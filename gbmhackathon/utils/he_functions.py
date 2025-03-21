import io
import pandas as pd
import s3fs
from gbmhackathon.definitions import MOSAIC_BUCKET

def get_tif_bytes_io(
    slide_path: pd.DataFrame, bucket: str = MOSAIC_BUCKET
):
    s3 = s3fs.S3FileSystem()
    s3_path = f"s3://{bucket}/{slide_path}"

    with s3.open(s3_path, "rb") as file_obj :
        bio = io.BytesIO(file_obj.read())
    return bio
