"""Module for loading data."""

import os

import io
import anndata as ad
import pandas as pd
import scanpy as sc
import boto3
from gbmhackathon.s3_loader import list_bucket_files


class BaseDataLoader:
    """Base class for loader."""

    def __init__(self, data_transformer=None, bucket = None, **load_data_kwargs):
        """Initialize base class.

        Parameters
        ----------
        data_transformer : optional
            An instance of DataCleaner or its subclass for data
            preprocessing, by default None
        **load_data_kwargs : dict
            Additional keyword arguments for the load_data method.
        """
        self.data_transformer = data_transformer
        self.load_data_kwargs = load_data_kwargs
        self.s3 = boto3.client("s3")
        self.bucket = bucket

    def load_data(self, data_path):
        """Load and transform the input data.

        Parameters
        ----------
        data_path : str
            Path to the data file to be read.

        Returns
        -------
        data : DataFrame
            Loaded and transformed data.
        """
        pass


class XLSXDataLoader(BaseDataLoader):
    """A class for loading Excel data.

    This transformer loads and preprocesses data using pandas.
    """

    def __init__(self, data_transformer=None, **load_data_kwargs):
        """Initialize the CSVDataLoader object.

        Parameters
        ----------
        data_transformer : optional
            An instance of DataCleaner for data
            preprocessing, default is None.
        load_data_kwargs : dict
            Additional keyword arguments for the load_data method.
        """
        # if data_transformer is None the data will be returned in the same
        # format as loaded

        super().__init__(data_transformer=data_transformer, **load_data_kwargs)

    def load_data(self, data_path):
        """Load and transform the input data.

        Parameters
        ----------
         data_path : str
            Path to the Excel file to be read.

        Returns
        -------
        X_csv : DataFrame
            Loaded and transformed data not including the target columns
        y : DataFrame
            Data corresponding to target columns with index as patient ids.
        """
        data = pd.read_excel(data_path, **self.load_data_kwargs)
        if self.data_transformer is not None:
            data = self.data_transformer.transform(data)
        return data


class CSVDataLoader(BaseDataLoader):
    """A class for loading CSV data.

    This transformer loads and preprocesses data using pandas.
    """

    def __init__(self, data_transformer=None, **load_data_kwargs):
        """Initialize the CSVDataLoader object.

        Parameters
        ----------
        dataset_name : str, optional
            Dataset name,
        data_transformer : optional
            An instance of DataCleaner or its subclass for data
            preprocessing, default is None.
        load_data_kwargs : dict
            Additional keyword arguments for the load_data method.
        """
        # if data_transformer is None the data will be returned in the same
        # format as loaded
        super().__init__(data_transformer=data_transformer, **load_data_kwargs)

    def load_data(
        self,
        data_path,
    ):
        """Transform the input data using pandas.

        Returns
        -------
        X_csv : DataFrame
            Loaded and transformed data not including the target columns
        y : DataFrame
            Data corresponding to target columns with index as patient ids.
        """
        with self.s3.get_object(Bucket = self.bucket, Key = str(data_path))["Body"] as file_obj : 
            data = pd.read_csv(filepath_or_buffer=file_obj, **self.load_data_kwargs)
            if self.data_transformer is not None:
                data = self.data_transformer.transform(data)
        return data


class PathsLoader(BaseDataLoader):
    """Class to load a data frame with the path to the paths of files in a directory."""

    def __init__(
        self, pattern_files: str = "*", data_transformer=None, **load_data_kwargs
    ):
        """Initialize the PathsLoader object.

        Parameters
        ----------
        pattern_files : str
            Pattern to match the files in the directory.
        data_transformer : optional
            An instance of DataCleaner for data
            preprocessing, default is None.
        load_data_kwargs : dict
            Additional keyword arguments for the load_data method.
        """
        super().__init__(data_transformer=data_transformer, **load_data_kwargs)
        self.pattern_files = pattern_files

    def load_data(self, data_path):
        """Load and transform the input data.

        Parameters
        ----------
        data_path : str
            Path to the folder containing the files to be included.

        Returns
        -------
        data : DataFrame
            Loaded and transformed data.
        """
        # paths = list(data_path.glob(self.pattern_files))
        paths = list_bucket_files(bucket_name=self.bucket, prefix=data_path,  pattern  = self.pattern_files)
        data = pd.DataFrame({"path": paths})
        if self.data_transformer is not None:
            data = self.data_transformer.transform(data)
        return data


class SingleCellLoader:
    """Class to load the single cell anndata."""
    def __init__(self, bucket = None) : 
        self.s3 = boto3.client("s3")
        self.bucket = bucket

    def load_data(self, path_data):
        """Load the single cell data.

        Parameters
        ----------
        path_data : str
            Path to the anndata object.

        Returns
        -------
        adata : anndata
            The anndata object.
        """

        with io.BytesIO(self.s3.get_object(Bucket = self.bucket, Key = str(path_data))["Body"].read()) as file_obj :
            adata = ad.read_h5ad(file_obj)
        return adata


class VisiumLoader:
    """Class to load the visium anndata."""

    def load_samples_to_anndata(
        self,
        input_path,
        sample_list,
        resolution,
    ):
        """Load AnnData objects from a folder containing one folder per sample using scanpy.read_visium.

        Parameters
        ----------
        input_path (str): Path to the root folder containing sample subfolders.
                        Each subfolder should contain a Visium dataset with spatial data.
        sample_list (Optional[List[str]]): List of sample IDs to load. If None, all samples in the folder are loaded.
        resolution (str): Resolution of the spatial image to load, either "hires" or "lowres". Default is "hires".

        Returns
        -------
        Dict[str, sc.AnnData]: A dictionary where keys are sample IDs (folder names) and
                            values are the loaded AnnData objects.
        """
        if resolution not in ["hires", "lowres"]:
            raise ValueError("Resolution must be either 'hires' or 'lowres'.")

        anndata_dict = {}

        # Get the list of samples to process
        samples_to_load = (
            sample_list if sample_list is not None else os.listdir(input_path)
        )

        # Iterate over the specified samples
        for sample_id in samples_to_load:
            sample_path = os.path.join(input_path, sample_id, "outs")

            if os.path.isdir(sample_path):
                try:
                    # Load the Visium dataset using scanpy.read_visium
                    adata = sc.read_visium(sample_path)
                    adata.var_names_make_unique()
                    # Extract the spatial image and ensure the desired resolution is available
                    library_id = list(adata.uns["spatial"].keys())[0]
                    if resolution in adata.uns["spatial"][library_id]["images"]:
                        adata.uns["spatial"][library_id]["use_for_plotting"] = (
                            resolution
                        )
                    else:
                        print(
                            f"Resolution '{resolution}' not available for sample {sample_id}. Skipping."
                        )
                        continue

                    # Add the AnnData object to the dictionary
                    anndata_dict[sample_id] = adata

                except Exception as e:
                    print(f"Failed to load data for sample {sample_id}: {e}")
            else:
                print(f"Sample directory not found: {sample_path}")

        return anndata_dict
