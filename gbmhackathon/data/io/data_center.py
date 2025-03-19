"""Define the DataCenter class."""

from __future__ import annotations

import functools
from typing import List, NamedTuple, Optional

from gbmhackathon.data.io.data_structure import DataFile, DataSource
from gbmhackathon.data.io.loaders import SingleCellLoader, VisiumLoader


CENTERS: dict[str, DataCenter] = {}


def register_center(cls):
    """Define decorator to register instances of DataCenter."""

    @functools.wraps(cls)
    def wrapper_register(*args, **kwargs):
        obj = cls(*args, **kwargs)
        CENTERS[obj.name] = obj
        return obj

    return wrapper_register


@register_center
class DataCenter(NamedTuple):
    """Define the structure of a data center."""

    name: str
    sources: dict[str, DataSource]

    def load_tabular(self):
        """Load tabular data.

        Returns
        -------
        dict[str, pd.DataFrame]
            Dictionary of dataframes for each source that are tabular.
        """
        return {
            source_name: {
                file_name: (file_.load())
                for file_name, file_ in source.files.items()  # Only call items() if source.files is a dict
            }
            if isinstance(source.files, dict)
            else source.files.load()
            # Load only the sources that are clinical, bulk_rna
            for source_name, source in self.sources.items()
            if source_name in ["metadata", "clinical", "bulk_rna", "wes", "he", "mibi_images"]
        }

    def load_singlecell(self, bucket):
        """Load single-cell data.

        Returns
        -------
        dict[str, pd.DataFrame]
            Dictionary of dataframes for each source that are single-cell.
        """
        file_path = self.sources["sc_rna"].files["scRNA anndata"].name
        return SingleCellLoader(bucket = bucket).load_data(file_path)

    def load_visium(
        self, sample_list: Optional[List[str]] = None, resolution: str = "lowres"
    ):
        """Load Visium data.

        Parameters
        ----------
        sample_list (Optional[List[str]]): List of sample IDs to load. If None, all samples in the folder are loaded.
        resolution (str): Resolution of the spatial image to load, either "hires" or "lowres". Default is "hires".

        Returns
        -------
        Dict[str, sc.AnnData]: A dictionary where keys are sample IDs (folder names) and
                               values are the loaded AnnData objects.
        """
        spatial_files = self.sources["spatial"].files
        if isinstance(spatial_files, dict):
            visium_file = spatial_files["Visium anndata"]
        else:
            raise TypeError("Expected a dictionary for 'spatial' files")
        if isinstance(visium_file, DataFile):
            file_path = visium_file.name
        else:
            raise TypeError("Expected a DataFile for 'Visium anndata'")
        if sample_list is None:
            print("No sample list provided. Loading all samples.")
        print("Resolution of the spatial image to load: ", resolution)
        print(
            "You can change the resolution by setting the resolution parameter using the resolution argument."
        )
        print("Loading Visium data, this can take few minutes...")
        return VisiumLoader().load_samples_to_anndata(
            file_path, sample_list, resolution
        )
