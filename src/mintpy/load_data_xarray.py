############################################################
# Program is part of MintPy                                #
# Copyright (c) 2013, Zhang Yunjun, Heresh Fattahi         #
# Author: Alex Lewandowski 2023                            #
############################################################

from pathlib import Path
from typing import Dict
import xarray as xr
import zarr

from mintpy import subset
from mintpy.defaults import auto_path
from mintpy.objects import sensor, geometry, ifgramStack
from mintpy.objects.coord import coordinate
from mintpy import load_data
from mintpy.objects.stackDict_xarray import geometryXarrayDict, ifgramStackXarrayDict
from mintpy.utils import ptime, readfile, utils, zarr_utils as ut
from mintpy.utils import utils1 as ut1

GEO_H5_PATH = Path.cwd()/"inputs/geometryGeo.h5"
SBAS_H5_PATH = Path.cwd()/"inputs/ifgramStack.h5"

IFG_XR_DSET_NAME2TEMPLATE_KEY = {
    'sbas_pair_list'  : 'mintpy.load.sbasPairList',
    'unwrapPhase'     : 'mintpy.load.unwVarName',
    'coherence'       : 'mintpy.load.corVarName',
}

GEOM_XR_DSET_NAME2TEMPLATE_KEY = {
    'height'          : 'mintpy.load.demVarName',
    'incidenceAngle'  : 'mintpy.load.incAngleVarName',
    'azimuthAngle'    : 'mintpy.load.azAngleVarName',
    'waterMask'       : 'mintpy.load.waterMaskVarName',
}

ZARR_NAME2TEMPLATE_KEY = {
    'zarr_s3_uri'     : 'mintpy.load.s3URI',
    'local_zarr_dir' : 'mintpy.load.localZarrDir',
    'aws_profile'     : 'mintpy.load.aws_profile',
    'zarr_group'      : 'mintpy.load.zarr_group',
}

def get_size(stack, box=None, xstep=1, ystep=1):
           # update due to subset
        if box:
            length = box[3] - box[1]
            width = box[2] - box[0]
        else:
            length = stack.shape[0]
            width = stack.shape[1]

        # update due to multilook
        length = length // ystep
        width = width // xstep

        return length, width

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
        pix_box, geo_box = subset.read_subset_template2box(iDict['template_file'])
        
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



def load_data_xarray(iDict):
    """load data into HDF5 files."""

    geo_group = "geometry" if iDict['mintpy.load.zarr_group'] == "auto" else f"{iDict['mintpy.load.zarr_group']}/geometry"
    sbas_group = "sbas" if iDict['mintpy.load.zarr_group'] == "auto" else f"{iDict['mintpy.load.zarr_group']}/sbas"
    if iDict['mintpy.load.s3URI'] == 'auto':
        try:
            geo_stack = ut.get_local_zarr_store(
                iDict['mintpy.load.localZarrDir'],
                geo_group
                )
        except zarr.errors.PathNotFoundError:
             print('"geometry" group not found in zarr store')
             pass
        try:
            sbas_stack = ut.get_local_zarr_store(
                iDict['mintpy.load.localZarrDir'],
                sbas_group
                )
        except zarr.errors.PathNotFoundError:
            print('"sbas" group not found in zarr store')
            pass
    else:
        try:
            geo_stack = ut.get_s3_zarr_store(
                iDict['mintpy.load.s3URI'],
                geo_group,
                iDict['mintpy.load.aws_profile']
                )
        except zarr.errors.PathNotFoundError:
             print('"geometry" group not found in zarr store')
             pass
        try:
            sbas_stack = ut.get_s3_zarr_store(
                iDict['mintpy.load.s3URI'],
                sbas_group,
                iDict['mintpy.load.aws_profile']
                )
        except zarr.errors.PathNotFoundError:
            print('"sbas" group not found in zarr store')
            pass

    ## search & write data files
    print('-'*50)
    print('updateMode : {}'.format(iDict['updateMode']))
    print('compression: {}'.format(iDict['compression']))
    print('multilook x/ystep: {}/{}'.format(iDict['xstep'], iDict['ystep']))
    print('multilook method : {}'.format(iDict['method']))
    kwargs = dict(xstep=iDict['xstep'], ystep=iDict['ystep'])

        # read subset info [need the metadata from above]
    iDict = load_data.read_subset_box(iDict)

    box = None
    if 'geo_box' in iDict.keys() and iDict['geo_box']:
        box = iDict['geo_box']
    elif 'pix_box' in iDict.keys() and iDict['pix_box']:
        box = iDict['pix_box']

    geo_dict = geometryXarrayDict(geo_stack, GEOM_XR_DSET_NAME2TEMPLATE_KEY, iDict)
    if load_data.run_or_skip(str(GEO_H5_PATH), geo_dict, geo_stack, box, **kwargs) == 'run':
        geo_dict.write2hdf5(
            GEO_H5_PATH, 
            access_mode='w',
            box=box,
            xstep=iDict['xstep'],
            ystep=iDict['ystep'],
            compression='lzf',
             )
    else:
        print("SKIPPING GEOMETRY")
        
    sbas_dict = ifgramStackXarrayDict(sbas_stack, IFG_XR_DSET_NAME2TEMPLATE_KEY, iDict)
    if load_data.run_or_skip(str(SBAS_H5_PATH), sbas_dict, sbas_stack, box, **kwargs) == 'run':
        sbas_dict.write2hdf5(
            SBAS_H5_PATH, 
            access_mode='w',
            box=box,
            xstep=iDict['xstep'],
            ystep=iDict['ystep'],
            compression='lzf',
             )
    else:
        print("SKIPPING")

    # sbas_dict.write2hdf5(box=box)
        
    

    # pass