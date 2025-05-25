import os
import pickle
from dotenv import load_dotenv
from pathlib import Path
import boto3
import re
import s3fs
from datetime import datetime

MOSAIC_DATASET = "ABSTRA_DATASET_03bb30aa_16ed_4b89_913e_fe009db2aabd"
BRUCE_DATESET = "ABSTRA_DATASET_8bfd41bf_a110_4748_bda1_8c225cdde6b5"
PROJECT_STORAGE = "ABSTRA_PROJECT_STORAGE_BUCKET"

def get_s3_dataset_info(dataset_name : str = MOSAIC_DATASET):
    load_dotenv()
    s3_link = os.environ.get(dataset_name)
    s3_bucket, s3_folder = s3_link.split("/")[2], Path(s3_link.split("/")[3])
    return s3_bucket, s3_folder

def list_bucket_files(bucket_name, prefix, pattern= "*" ):

    s3 = boto3.client("s3")
    regex = re.compile(pattern)

    try:
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix = str(prefix))

        if "Contents" in response:
            fichiers = [Path(obj["Key"]) for obj in response["Contents"] if regex.match(obj["Key"])]
            return fichiers
        else:
            # No file matching the pattern found
            return []

    except Exception as e:
        print(f"Error : {e}")
        return []

def ls_folder_bucket(folder : str, bucket):
    fs = s3fs.S3FileSystem()
    s3_path = f"s3://{bucket}/{folder}/" if folder[-1] != "/" else f"s3://{bucket}/{folder}"
    return [Path(folder).name for folder in fs.ls(s3_path)]

def load_visium_folder(sample_id : str,
                       s3_folder : str,
                       local_path : str,
                       bucket,
                       ):
    try :
        fs = s3fs.S3FileSystem()
        s3_path = f"s3://{bucket}/{s3_folder}/{sample_id}/"
        print(f"Copying S3 : {s3_path} to {local_path}")
        fs.get(s3_path, local_path, recursive=True)
        return (True, None)
    except Exception as e :
        print(f"Error to copy files : {e}")
        return (False, e)

def write_s3(obj, save_name : str, folder : str, bucket_name = PROJECT_STORAGE):
    try : 
        s3_bucket , s3_folder = get_s3_dataset_info(bucket_name)
        date = datetime.now().strftime("%Y-%m-%d_%H-%M")
        save_name = f"{folder}/{date}_{save_name}.pkl"
        path = f"s3://{s3_bucket}/{s3_folder}/{save_name}"
        fs = s3fs.S3FileSystem()
        with fs.open(path=path, mode = "wb") as f : 
            pickle.dump(obj, f)
        print(f"Object saved at {path}")
    except Exception as e :
        print(f"Failed to save object : {e}")
    
def load_s3(s3_path : str) :
    try : 
        fs = s3fs.S3FileSystem()
        with fs.open(s3_path, "rb") as f :
            obj = pickle.load(f)
        return obj
    except Exception as e :
        raise e
        print(f"Error to load {s3_path} : {e}")