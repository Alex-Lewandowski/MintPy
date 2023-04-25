############################################################
# Program is part of MintPy                                #
# Copyright (c) 2013, Zhang Yunjun, Heresh Fattahi         #
# Author: Heresh Fattahi, Zhang Yunjun, 2017               #
############################################################
# class used for data loading from InSAR stack to MintPy timeseries
# Recommend import:
#     from mintpy.objects.stackDict import (geometryXarrayDict,
#                                           ifgramStackXarrayDict)

import os
import time
import warnings

import h5py
import numpy as np
from skimage.transform import resize
import xarray as xr

from mintpy.multilook import multilook_data
from mintpy.objects import (
    DATA_TYPE_DICT,
    GEOMETRY_DSET_NAMES,
    IFGRAM_DSET_NAMES,
)
from mintpy.utils import attribute as attr, ptime, readfile, utils0 as ut

from typing import Dict
from mintpy.objects.coord import coordinate
from mintpy import subset


def read_subset_box_xarray(iDict: Dict, stack: xr.Dataset):
        """Reads mintpy.subset.yx and mintpy.subset.lalo from a template file and
        Updates iDict with a 'geo_box' and 'pix_box' if they are contained by the Dataset
        
        Parameters:
        iDict: dictionary containing a key 'template_file' with a path to a mintpy template file
        stack: an xarray.Dataset with 'x' and 'y' dimensions (geo_box must be provided in same CRS as Dataset)
        
        Returns:
        iDict (updated with bboxes)
        """
        # Read subset info from template
        template_path = [i for i in iDict['template_file'] if "smallbaselineApp.cfg" in i]
        print(template_path)

        pix_box, geo_box = subset.read_subset_template2box(template_path[0])
        
        iDict['geo_box'] = None
        iDict['pix_box'] = None        
        if geo_box:
            # confirm bbox in Dataset
            if stack.coords['x'][0] <= geo_box[0] <= geo_box[2] <= stack.coords['x'][-1] \
            and stack.coords['y'][-1] <= geo_box[3] <= geo_box[1] <= stack.coords['y'][0]:
                iDict['geo_box'] = geo_box
        if pix_box:
            if 0 <= pix_box[0] < pix_box[2] < len(stack.coords['x']) \
            and 0 <= pix_box[1] < pix_box[3] < len(stack.coords['y']):
                iDict['pix_box'] = pix_box                                           
        return iDict


class geometryXarrayDict:
    """
    Geometry object for Height, Incidence Angles, Water Masks, ... from the same platform and track
    """

    def __init__(self, stack: xr.Dataset, datasetDict, iDict: Dict, name: str='geometry'):
        self.name = name
        self.datasetDict = datasetDict
        self.stack = stack
        self.iDict = iDict

        meta_vars = [i for i in stack.variables if i not in {k:v for (k,v) in zip(iDict.keys(), iDict.values()) if k in datasetDict.values()}.values() and i not in ['x', 'y']]
        self.metadata = {}
        for m in meta_vars:
            self.metadata[m] = stack[m].to_numpy().tolist()

    def read(self, dsName, box=None, xstep=1, ystep=1):
        data = self.stack[self.datasetDict[dsName]].to_numpy()
        
        return data, self.metadata
    
    def get_size(self, dsName=None, box=None, xstep=1, ystep=1):
           # update due to subset
        if box:
            length = box[3] - box[1]
            width = box[2] - box[0]
        else:
            length = int(self.metadata['LENGTH'])
            width = int(self.metadata['WIDTH'])

        # update due to multilook
        length = length // ystep
        width = width // xstep

        return length, width
       
    def write2hdf5(self, outputFile='geometryRadar.h5', access_mode='w', box=None, xstep=1, ystep=1,
                   compression='lzf'):
        """Save/write to HDF5 file with structure defined in:
            https://mintpy.readthedocs.io/en/latest/api/data_structure/#geometry
        """
        print('-'*50)
        if len(self.datasetDict) == 0:
            print('No xarray.Dataset Variable names in the object, skip HDF5 file writing.')
            return None

        # output directory
        output_dir = os.path.dirname(outputFile)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
            print(f'create directory: {output_dir}')

        maxDigit = max(len(i) for i in GEOMETRY_DSET_NAMES)
        length, width = self.get_size(box=box, xstep=xstep, ystep=ystep)

        self.outputFile = outputFile
        with h5py.File(self.outputFile, access_mode) as f:
            print(f'create HDF5 file {self.outputFile} with {access_mode} mode')

            ###############################
            for dsName in self.datasetDict.keys():
                if self.iDict[self.datasetDict[dsName]] == 'auto':
                    continue
                
                dsDataType = np.float32
                if dsName.lower().endswith('mask'):
                    dsDataType = np.bool_
                dsShape = (length, width)
                print((f'create dataset /{dsName:<{maxDigit}} of {str(dsDataType):<25} in size of {dsShape} with compression = {str(compression)}'))
       
                # retrieve subset bboxes from template
                self.iDict = read_subset_box_xarray(self.iDict, self.stack)
                      
                bbox = None
                # read data
                if self.iDict['geo_box']:
                    bbox = self.iDict['geo_box']
                    data = self.stack.sel(x=slice(bbox[0], bbox[2]), y=slice(bbox[1], bbox[3]))[self.iDict[self.datasetDict[dsName]]].to_numpy()
                elif self.iDict['pix_box']:
                    bbox = self.iDict['pix_box']
                    data = self.stack.isel(x=slice(bbox[0], bbox[2]), y=slice(bbox[1], bbox[3]))[self.iDict[self.datasetDict[dsName]]].to_numpy()
                else:
                    data = self.stack[self.iDict[self.datasetDict[dsName]]].to_numpy()
                    

                if dsName == 'height':
                    noDataValueDEM = -32768
                    if np.any(data == noDataValueDEM):
                        data[data == noDataValueDEM] = np.nan
                        print(f'    convert no-data value for DEM {noDataValueDEM} to NaN.')

                elif dsName == 'rangeCoord' and xstep != 1:
                    print(f'    scale value of {dsName:<15} by 1/{xstep} due to multilooking')
                    data /= xstep

                elif dsName == 'azimuthCoord' and ystep != 1:
                    print(f'    scale value of {dsName:<15} by 1/{ystep} due to multilooking')
                    data /= ystep

                elif dsName in ['incidenceAngle', 'azimuthAngle']:
                    # HyP3 (Gamma) angle of the line-of-sight vector (from ground to SAR platform)
                    # incidence angle 'theta' is measured from horizontal in radians
                    # azimuth   angle 'phi'   is measured from the east with anti-clockwise as positivve in radians               

                    if self.metadata.get('PROCESSOR', 'isce') == 'hyp3' and self.metadata.get('UNIT', 'degrees').startswith('rad'):

                        if dsName == 'incidenceAngle':
                            msg = f'    convert {dsName:<15} from Gamma (from horizontal in radian) '
                            msg += ' to MintPy (from vertical in degree) convention.'
                            print(msg)
                            data[data == 0] = np.nan                        # convert no-data-value from 0 to nan
                            data = 90. - (data * 180. / np.pi)              # hyp3/gamma to mintpy/isce2 convention
                            
                            print(f"incidenceAngle: {data}")

                        elif dsName == 'azimuthAngle':
                            msg = f'    convert {dsName:<15} from Gamma (from east in radian) '
                            msg += ' to MintPy (from north in degree) convention.'
                            print(msg)
                            data[data == 0] = np.nan                        # convert no-data-value from 0 to nan
                            data = data * 180. / np.pi - 90.                # hyp3/gamma to mintpy/isce2 convention
                            data = ut.wrap(data, wrap_range=[-180, 180])    # rewrap within -180 to 180

                # write
                ds = f.create_dataset(dsName,
                                      data=data,
                                      chunks=True,
                                      compression=compression)

            # update due to subset
            self.metadata = attr.update_attribute4subset(self.metadata, box)
            # update due to multilook
            if xstep * ystep > 1:
                self.metadata = attr.update_attribute4multilook(self.metadata, ystep, xstep)

            self.metadata['FILE_TYPE'] = self.name
            for key, value in self.metadata.items():
                if type(value) in [str, float, int]:
                    f.attrs[key] = value

        print(f'Finished writing to {self.outputFile}')
        return self.outputFile