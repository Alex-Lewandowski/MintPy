############################################################
# Program is part of MintPy                                #
# Copyright (c) 2013, Zhang Yunjun, Heresh Fattahi         #
# Author: Alex Lewandowski 2023                            #
############################################################

# class used for data loading from InSAR stack to MintPy timeseries
# Recommend import:
#     from mintpy.objects.stackDict import (geometryXarrayDict,
#                                           ifgramStackXarrayDict)

import os
from pathlib import Path
import shutil
import sys
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
        # retrieve subset bboxes from template
        self.iDict = read_subset_box_xarray(self.iDict, self.stack)

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
    
    def get_slant_range_distance(self, box=None, xstep=1, ystep=1):
        """Generate 2D slant range distance if missing from input template file"""
        print('prepare slantRangeDistance ...')
        if 'Y_FIRST' in self.metadata.keys():
            # for dataset in geo-coordinates, use:
            # 1) incidenceAngle matrix if available OR
            # 2) contant value from SLANT_RANGE_DISTANCE.
            ds_name = 'incidenceAngle'
            key = 'SLANT_RANGE_DISTANCE'
            if ds_name in self.datasetDict.keys():
                print(f'    geocoded input, use incidenceAngle from xarray.Dataset variable: {self.iDict[self.datasetDict[ds_name]]}')
                inc_angle = self.stack[self.iDict[self.datasetDict[ds_name]]].to_numpy()
                if self.metadata.get('PROCESSOR', 'isce') == 'hyp3_zarr' and self.metadata.get('UNIT', 'degrees').startswith('rad'):
                    print('    convert incidence angle from Gamma to MintPy convention.')
                    inc_angle[inc_angle == 0] = np.nan # convert the no-data-value from 0 to nan         
                    inc_angle = 90. - (inc_angle * 180. / np.pi)    # hyp3/gamma to mintpy/isce2 convention
                # inc angle -> slant range distance
                data = ut.incidence_angle2slant_range_distance(self.metadata, inc_angle)

            elif key in self.metadata.keys():
                print(f'geocoded input, use contant value from metadata {key}')
                length = int(self.metadata['LENGTH'])
                width = int(self.metadata['WIDTH'])
                range_dist = float(self.metadata[key])
                data = np.ones((length, width), dtype=np.float32) * range_dist
            else:
                return None

        else:
            # for dataset in radar-coordinates, calculate 2D pixel-wise value from geometry
            data = ut.range_distance(self.metadata,
                                     dimension=2,
                                     print_msg=False)

        # subset
        if box is not None:
            data = data[box[1]:box[3],
                        box[0]:box[2]]

        # multilook
        if xstep * ystep > 1:
            # output size if x/ystep > 1
            xsize = int(data.shape[1] / xstep)
            ysize = int(data.shape[0] / ystep)

            # sampling
            data = data[int(ystep/2)::ystep,
                        int(xstep/2)::xstep]
            data = data[:ysize, :xsize]

        return data
       
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
        output_dir = Path(outputFile).parent
        try:
            output_dir.mkdir()
            print(f'create directory: {output_dir}')
        except FileExistsError:
            pass

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
                

            ###############################
            # Generate Dataset if not existed in binary file: incidenceAngle, slantRangeDistance
            
            
            print(self.datasetDict.keys())
            
            
            if 'slantRangeDistance' not in self.datasetDict.keys() or self.iDict[self.datasetDict['slantRangeDistance']] != 'auto':
                # Calculate data
                data = self.get_slant_range_distance(box=box, xstep=xstep, ystep=ystep)

                # Write dataset
                if data is not None:
                    dsShape = data.shape
                    dsDataType = np.float32
                    print(('create dataset /{d:<{w}} of {t:<25} in size of {s}'
                           ' with compression = {c}').format(d=dsName,
                                                             w=maxDigit,
                                                             t=str(dsDataType),
                                                             s=dsShape,
                                                             c=str(compression)))
                    ds = f.create_dataset(dsName,
                                          data=data,
                                          dtype=dsDataType,
                                          chunks=True,
                                          compression=compression)

            ###############################

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
    

class ifgramStackXarrayDict:
    """
    IfgramStack object for a set of InSAR pairs from the same platform and track.
    """

    def __init__(self, stack: xr.Dataset, datasetDict, iDict: Dict, name: str='ifgramStack'):
        self.name = name
        self.datasetDict = datasetDict
        self.stack = stack
        
        # retrieve subset bboxes from template
        self.iDict = iDict
        self.iDict = read_subset_box_xarray(self.iDict, self.stack)

        self.sbas_pairs = iDict['mintpy.load.sbasPairList']
        if self.sbas_pairs == 'auto':
            self.sbas_pairs = self.stack.pairs.to_numpy()

        self.ds_vars = {k:v for (k,v) in zip(iDict.keys(), iDict.values()) if k in datasetDict.values()}
          
        meta_vars = [i for i in stack.variables if i not in {k:v for (k,v) in zip(iDict.keys(), iDict.values()) if k in datasetDict.values()}.values() and i not in ['x', 'y']]
        self.metadata = {}
        for m in meta_vars:
            self.metadata[m] = stack.isel(pairs=0)[m].to_numpy().tolist()

    def get_size(self, dsName, box=None, xstep=1, ystep=1, geom_obj=None):
        """Get size in 3D"""
        length, width = self.stack.isel(pairs=0)[dsName].shape

        # use the reference geometry obj size
        # for low-reso ionosphere from isce2/topsStack
        if geom_obj:
            length, width = geom_obj.get_size()

        # update due to subset
        if box:
            length, width = box[3] - box[1], box[2] - box[0]

        # update due to multilook
        length = length // ystep
        width = width // xstep

    def get_perp_baseline(self, date_pair):
        bperp_top = float(self.stack.sel(pairs=date_pair)['P_BASELINE_TOP_HDR'].to_numpy())
        bperp_bottom = float(self.stack.sel(pairs=date_pair)['P_BASELINE_BOTTOM_HDR'].to_numpy())
        return (bperp_top + bperp_bottom) / 2.0

    def write2hdf5(self, outputFile='ifgramStack.h5', access_mode='w', box=None, xstep=1, ystep=1, mli_method='nearest',
                   compression='lzf', geom_obj=None):
        """Save/write an ifgramStackDict object into an HDF5 file with the structure defined in:

        https://mintpy.readthedocs.io/en/latest/api/data_structure/#ifgramstack

        Parameters: outputFile     - str, Name of the HDF5 file for the InSAR stack
                    access_mode    - str, access mode of output File, e.g. w, r
                    box            - tuple, subset range in (x0, y0, x1, y1)
                    x/ystep        - int, multilook number in x/y direction
                    mli_method     - str, multilook method, nearest, mean or median
                    compression    - str, HDF5 dataset compression method, None, lzf or gzip
                    extra_metadata - dict, extra metadata to be added into output file
                    geom_obj       - geometryDict object, size reference to determine the resizing operation.
        Returns:    outputFile     - str, Name of the HDF5 file for the InSAR stack
        """
        print('-'*50)

                # output directory
        output_dir = Path(outputFile).parent
        try:
            output_dir.mkdir()
            print(f'create directory: {output_dir}')
        except FileExistsError:
            pass

        # used for formatting strings
        maxDigit = max(len(i) for i in self.ds_vars)

        # write HDF5 file
        with h5py.File(outputFile, access_mode) as f:
            print(f'create HDF5 file {outputFile} with {access_mode} mode')

            ###############################
            # 3D datasets containing unwrapPhase, coherence, waterMask(optional)
            for dsName in self.ds_vars:
                if dsName != 'mintpy.load.sbasPairList':
                    if dsName == 'mintpy.load.unwVarName':
                        dsName_out = 'unwrapPhase'
                    elif dsName == 'mintpy.load.corVarName':
                        dsName_out = 'coherence'
                
           
                    print(f"dsName: {dsName}")
                    # print(f"iDict[dsName]: {self.iDict[dsName]}")



                    dsShape = (
                        len(self.sbas_pairs),
                        self.stack[self.iDict[dsName]].shape[1], 
                        self.stack[self.iDict[dsName]].shape[2]
                        )
                    dsDataType = np.float32
                    dsCompression = compression


                    print((f'create dataset /{dsName:<{maxDigit}} of {str(dsDataType):<25} in size of {dsShape}'
                        ' with compression = {dsCompression}'))
                    ds = f.create_dataset(dsName_out,
                                        shape=dsShape,
                                        maxshape=(None, dsShape[1], dsShape[2]),
                                        dtype=dsDataType,
                                        chunks=True,
                                        compression=dsCompression)

                    prog_bar = ptime.progressBar(maxValue=dsShape[0])
                    for i, pair in enumerate(self.sbas_pairs):
                        prog_bar.update(i+1, suffix=f'{pair}')

                        # read data 
                        if self.iDict['geo_box']:
                            bbox = self.iDict['geo_box']
                            data = self.stack.sel(pairs=pair, x=slice(bbox[0], bbox[2]), y=slice(bbox[1], bbox[3]))[self.iDict[dsName]].to_numpy()
                        elif self.iDict['pix_box']:
                            bbox = self.iDict['pix_box']
                            data = self.stack.isel(pairs=0, x=slice(bbox[0], bbox[2]), y=slice(bbox[1], bbox[3]))[self.iDict[dsName]].to_numpy()
                        else:
                            data = self.stack.sel(pairs=pair)[self.iDict[dsName]].to_numpy()

                        # multilook
                        if xstep * ystep > 1:
                            if mli_method == 'nearest':
                                # multilook - nearest resampling
                                # output data size
                                xsize = int(data.shape[1] / xstep)
                                ysize = int(data.shape[0] / ystep)
                                # sampling
                                data = data[int(ystep/2)::ystep,
                                            int(xstep/2)::xstep]
                                data = data[:ysize, :xsize]

                            else:
                                # multilook - mean or median resampling
                                data = multilook_data(data,
                                                    lks_y=ystep,
                                                    lks_x=xstep,
                                                    method=mli_method)

                        # write
                        ds[i, :, :] = data
                        

  
                    ds.attrs['WIDTH'] = ds[0].shape[1]
                    ds.attrs['LENGTH'] = ds[0].shape[0]
                    ds.attrs['MODIFICATION_TIME'] = str(time.time())
                    
                    
                    # ds.attrs['X_FIRST'] = self.metadata['X_FIRST'][0]
                    # ds.attrs['Y_FIRST'] = self.metadata['Y_FIRST'][0]
                    # ds.attrs['X_STEP'] = self.metadata['X_STEP'][0]
                    # ds.attrs['Y_STEP'] = self.metadata['Y_STEP'][0]
                    # ds.attrs['X_UNIT'] = self.metadata['X_UNIT'][0]
                    # ds.attrs['Y_UNIT'] = self.metadata['Y_UNIT'][0]

                    
                    prog_bar.close()

            #############################
            # 2D dataset containing reference and secondary dates of all pairs
            dsName = 'date'
            dsDataType = np.string_
            print(f'create dataset /{dsName:<{maxDigit}} of {str(dsDataType):<25} in size of {self.sbas_pairs.shape}')
            sbas_pairs = [i.split('-') for i in self.sbas_pairs]
            f.create_dataset(dsName, data=np.array(sbas_pairs, dtype=dsDataType))

            ###############################
            # 1D dataset containing perpendicular baseline of all pairs
            dsName = 'bperp'
            dsDataType = np.float32
            data = np.array([self.get_perp_baseline(b) for b in self.sbas_pairs], dtype=dsDataType)
            print(f'create dataset /{dsName:<{maxDigit}} of {str(dsDataType):<25} in size of {data.shape}')
            f.create_dataset(dsName, data=data)

            ###############################
            # 1D dataset containing bool value of dropping the interferograms or not
            dsName = 'dropIfgram'
            dsDataType = np.bool_
            dsShape = (len(self.sbas_pairs),)
            print(f'create dataset /{dsName:<{maxDigit}} of {str(dsDataType):<25} in size of {dsShape}')
            data = np.ones(dsShape, dtype=dsDataType)
            f.create_dataset(dsName, data=data)

            ###############################
            # Attributes

            # update metadata due to subset
            if box:
                print('update metadata due to subset')
                self.metadata = attr.update_attribute4subset(self.metadata, box)

            # update metadata due to multilook
            if xstep * ystep > 1:
                print('update metadata due to multilook')
                self.metadata = attr.update_attribute4multilook(self.metadata, ystep, xstep)

            # write metadata to HDF5 file at the root level
            self.metadata['FILE_TYPE'] = self.name
            
            print(self.metadata)
            
            for key, value in self.metadata.items():
                if type(value) in [str, float, int]:
                    f.attrs[key] = value
                else:
                    f.attrs[key] = value[0]

        print(f'Finished writing to {outputFile}')
        return outputFile

