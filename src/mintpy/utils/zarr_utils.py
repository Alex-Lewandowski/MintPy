############################################################
# Program is part of MintPy                                #
# Copyright (c) 2013, Zhang Yunjun, Heresh Fattahi         #
# Author: Zhang Yunjun, Heresh Fattahi, 2013               #
############################################################

import s3fs
import xarray as xr

def get_s3_zarr_store(s3_uri: str, group: str, profile: str='default') -> xr.Dataset:
    """
    Open a Zarr store in an AWS S3 bucket.
    Assumes that you have aws-cli configured with an 
    API key for an IAM user with access to the S3 bucket.

    Parameters:
    s3_uri: a string of the S3 bucket's URI
    group: a string to the group containing the Zarr Store
    profile: a string of aws profile to use for authentication

    Returns:
        an xarray.Dataset of zarr store
    """

    s3 = s3fs.S3FileSystem(profile=profile)
    store = s3fs.S3Map(root=s3_uri, s3=s3, check=False)
    return xr.open_zarr(store=store, group=group)

   