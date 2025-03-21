import io
import gc
import tiffslide
import pandas as pd
import boto3
from gbmhackathon.definitions import MOSAIC_BUCKET

def get_tiff_path(
    slide_paths: pd.DataFrame, slide_idx: int, bucket: str = MOSAIC_BUCKET
):
    s3 = boto3.client("s3")
    slide_path = str(slide_paths["path"].iloc[slide_idx])
    buffer = io.BytesIO(s3.get_object(Bucket=bucket, Key=slide_path)["Body"].read())
    with buffer as file_obj:
        slide = tiffslide.TiffSlide(file_obj)
    del buffer
    gc.collect()
    return slide
