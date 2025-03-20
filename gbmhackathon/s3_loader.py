import os
from dotenv import load_dotenv
from pathlib import Path
import boto3
import re

MOSAIC_DATASET = "ABSTRA_DATASET_03bb30aa_16ed_4b89_913e_fe009db2aabd"
BRUCE_DATESET = "ABSTRA_DATASET_8bfd41bf_a110_4748_bda1_8c225cdde6b5"

def get_s3_dataset_info(dataset_name : str = MOSAIC_DATASET):
    load_dotenv()
    s3_link = os.environ.get(dataset_name)
    s3_bucket, s3_folder = s3_link.split("/")[2], Path(s3_link.split("/")[3])
    return s3_bucket, s3_folder

def list_bucket_files(bucket_name, pattern= "*" ):

    s3 = boto3.client("s3")
    regex = re.compile(pattern)

    try:
        response = s3.list_objects_v2(Bucket=bucket_name)

        if "Contents" in response:
            fichiers = [obj["Key"] for obj in response["Contents"] if regex.match(obj["Key"])]
            return fichiers
        else:
            # No file matching the pattern found
            return []

    except Exception as e:
        print(f"Error : {e}")
        return []
