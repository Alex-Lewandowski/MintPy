from pathlib import Path
from typing import Dict
import xarray as xr

from mintpy import subset
from mintpy.defaults import auto_path
from mintpy.objects import sensor, geometry, ifgramStack
from mintpy.objects.coord import coordinate
from mintpy import load_data
from mintpy.objects.stackDict_xarray import geometryXarrayDict
from mintpy.utils import ptime, readfile, utils, zarr_utils as ut

GEO_H5_PATH = Path.cwd()/"inputs/geometryGeo.h5"

IFG_XR_DSET_NAME2TEMPLATE_KEY = {
    'sbas_pair_list'  : 'mintpy.load.sbasPairList',
    'ifgram_pairs'    : 'mintpy.load.ifgramPairCoord',
    'unwrapPhase'     : 'mintpy.load.unwVarName',
    'coherence'       : 'mintpy.load.corVarName',
    # 'connectComponent': 'mintpy.load.connCompVarName',
    # 'wrapPhase'       : 'mintpy.load.intVarName',
    # 'magnitude'       : 'mintpy.load.magVarName',
}

GEOM_XR_DSET_NAME2TEMPLATE_KEY = {
    'height'          : 'mintpy.load.demVarName',
    'incidenceAngle'  : 'mintpy.load.incAngleVarName',
    'azimuthAngle'    : 'mintpy.load.azAngleVarName',
    'waterMask'       : 'mintpy.load.waterMaskVarName',
}

###
# TODO: add local zarr store path params
ZARR_NAME2TEMPLATE_KEY = {
    'zarr_s3_uri'     : 'mintpy.load.s3URI',
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

def run_or_skip(out_path, stack, ds_type, iDict, xstep=1, ystep=1):
    """Check if re-writing is necessary.
    Do not write HDF5 file if ALL the following meet:
        1. HDF5 file exists and is readable,
        2. HDF5 file constains all the datasets and in the same size
        3. For ifgramStackDict, HDF5 file contains all date12.
    Parameters: outFile    - str, path to the output HDF5 file
                inObj      - ifgramStackDict or geometryDict, object to write
                box        - tuple of int, bounding box in (x0, y0, x1, y1)
                updateMode - bool
                x/ystep    - int
                geom_obj   - geometryDict object or None, for ionosphere only
    Returns:    flag       - str, run or skip
    """
    box = None
    if iDict['geo_box']:
        box = iDict['geo_box']
    elif iDict['pix_box']:
        box = iDict['pix_box']    
        
    if ut.run_or_skip(out_path, readable=True) == 'skip':
        kwargs = dict(box=box, xstep=xstep, ystep=ystep)

        if ds_type == 'ifgramStack':
            in_size = get_size(stack, **kwargs)[1:]
            in_dset_list = inObj.get_dataset_list()
            in_date12_list = inObj.get_date12_list()

            outObj = ifgramStack(out_path)
            outObj.open(print_msg=False)
            out_size = (outObj.length, outObj.width)
            out_dset_list = outObj.datasetNames
            out_date12_list = outObj.date12List

            if (out_size[1:] == in_size[1:]
                    and set(in_dset_list).issubset(set(out_dset_list))
                    and set(in_date12_list).issubset(set(out_date12_list))):
                print('All date12   exists in file {} with same size as required,'
                      ' no need to re-load.'.format(os.path.basename(out_path)))
                flag = 'skip'

        elif ds_type == 'geometry':
            in_size = inObj.get_size(**kwargs)
            in_dset_list = inObj.get_dataset_list()

            outObj = geometry(out_path)
            outObj.open(print_msg=False)
            out_size = (outObj.length, outObj.width)
            out_dset_list = outObj.datasetNames

            if (out_size == in_size
                    and set(in_dset_list).issubset(set(out_dset_list))):
                print('All datasets exists in file {} with same size as required,'
                      ' no need to re-load.'.format(os.path.basename(out_path)))
                flag = 'skip'

    return flag

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

    stack = ut.get_s3_zarr_store(
        iDict['mintpy.load.s3URI'],
        iDict['mintpy.load.zarr_group'],
        iDict['mintpy.load.aws_profile']
          )

    ## search & write data files
    print('-'*50)
    print('updateMode : {}'.format(iDict['updateMode']))
    print('compression: {}'.format(iDict['compression']))
    print('multilook x/ystep: {}/{}'.format(iDict['xstep'], iDict['ystep']))
    print('multilook method : {}'.format(iDict['method']))
    kwargs = dict(updateMode=iDict['updateMode'], xstep=iDict['xstep'], ystep=iDict['ystep'])

        # read subset info [need the metadata from above]
    iDict = load_data.read_subset_box(iDict)

       # geometry in geo / radar coordinates
    geom_dset_name2template_key = {
        **GEOM_XR_DSET_NAME2TEMPLATE_KEY,
        **IFG_XR_DSET_NAME2TEMPLATE_KEY,
    }

    if iDict['geo_box']:
        box = iDict['geo_box']
    elif iDict['pix_box']:
        box = iDict['pix_box']

    for i in iDict:
        print(iDict)

    print("yup")

    # geo_dict = geometryXarrayDict(stack, )
    # if run_or_skip(GEO_H5_PATH, , box, **kwargs):
    #     pass

    # pass