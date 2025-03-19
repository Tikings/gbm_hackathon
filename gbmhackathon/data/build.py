"""A module for building the data centers loaders."""

from __future__ import annotations

from attrs import frozen

from gbmhackathon.data.io.data_center import DataCenter
from gbmhackathon.data.io.data_structure import (
    DataFile,
    DataSource,
)
from gbmhackathon.data.io.loaders import (
    CSVDataLoader,
    PathsLoader,
)
from gbmhackathon.data.io.preprocessor_tabular import (
    MOSAICBulkRNADataCleaner,
    MOSAICClinicalDataCleaner,
    MOSAICHEDataCleaner,
    MOSAICHEFeaturesDataCleaner,
    MOSAICKeyEventsDataCleaner,
    MOSAICLightClinicalDataCleaner,
    MOSAICWESDataCleaner,
    BruceLightMetadataDataCleaner,
    BruceMIBIImagesCleaner
)
from gbmhackathon.definitions import (
    DATA_PATH_FILES_MOSAIC,
    DATA_PATH_FILES_BRUCE,
    MOSAIC_BUCKET,
    BRUCE_BUCKET
)


@frozen
class SourceNames:
    """Define the names of the sources.

    This class is used to avoid typos when accessing the sources of a center.
    """

    metadata: str = "metadata"
    clinical: str = "clinical"
    wes: str = "wes"
    bulk_rna: str = "bulk_rna"
    sc_rna: str = "sc_rna"
    spatial: str = "spatial"
    he: str = "he"
    mibi_images: str = "mibi_images"

    # force default values by forbidding arguments during initialization.
    def __init__(self):
        self.__attrs_init__()  # type: ignore # pylint: disable=no-member


SOURCES = SourceNames()

MosaicDataset = DataCenter(
    name="mosaic",
    sources={
        SOURCES.clinical: DataSource(
            files={
                "data dictionary": DataFile(
                    name=DATA_PATH_FILES_MOSAIC["Data dictionary"],
                    loader=CSVDataLoader(
                        bucket=MOSAIC_BUCKET,
                        ),
                ),
                "original clinical": DataFile(
                    name=DATA_PATH_FILES_MOSAIC["Processed clinical"],
                    loader=CSVDataLoader(
                        bucket=MOSAIC_BUCKET,
                        data_transformer=MOSAICLightClinicalDataCleaner(),
                    ),
                ),
                "processed gbm clinical": DataFile(
                    name=DATA_PATH_FILES_MOSAIC["Processed clinical"],
                    loader=CSVDataLoader(
                        bucket=MOSAIC_BUCKET,
                        data_transformer=MOSAICClinicalDataCleaner(),
                    ),
                ),
                "treatments": DataFile(
                    name=DATA_PATH_FILES_MOSAIC["Treatments"],
                    loader=CSVDataLoader(
                        bucket=MOSAIC_BUCKET,
                        ),
                ),
                "key events clinical": DataFile(
                    name=DATA_PATH_FILES_MOSAIC["Key events clinical"],
                    loader=CSVDataLoader(
                        bucket=MOSAIC_BUCKET,
                        data_transformer=MOSAICKeyEventsDataCleaner(),
                    ),
                ),
            }
        ),
        SOURCES.bulk_rna: DataSource(
            files={
                "raw counts": DataFile(
                    name=DATA_PATH_FILES_MOSAIC["Bulk RNA raw counts"],
                    loader=CSVDataLoader(
                        bucket=MOSAIC_BUCKET,
                        data_transformer=MOSAICBulkRNADataCleaner(), sep="\t"
                    ),
                ),
                "TPM counts": DataFile(
                    name=DATA_PATH_FILES_MOSAIC["Bulk RNA TPM counts"],
                    loader=CSVDataLoader(
                        bucket=MOSAIC_BUCKET,
                        data_transformer=MOSAICBulkRNADataCleaner(), sep="\t"
                    ),
                ),
                "normalized counts": DataFile(
                    name=DATA_PATH_FILES_MOSAIC["Bulk RNA normalized counts"],
                    loader=CSVDataLoader(
                        bucket=MOSAIC_BUCKET,
                        data_transformer=MOSAICBulkRNADataCleaner(), sep="\t"
                    ),
                ),
                "fpkm counts": DataFile(
                    name=DATA_PATH_FILES_MOSAIC["Bulk RNA fpkm counts"],
                    loader=CSVDataLoader(
                        bucket=MOSAIC_BUCKET,
                        data_transformer=MOSAICBulkRNADataCleaner(), sep="\t"
                    ),
                ),
            }
        ),
        SOURCES.spatial: DataSource(
            files={
                "Visium anndata": DataFile(
                    name=DATA_PATH_FILES_MOSAIC["Visium anndata"],
                    loader=None,
                ),
            }
        ),
        SOURCES.sc_rna: DataSource(
            files={
                "scRNA anndata": DataFile(
                    name=DATA_PATH_FILES_MOSAIC["Single cell anndata"],
                    loader=None,
                ),
            }
        ),
        SOURCES.wes: DataSource(
            files={
                "WES CNV deletion": DataFile(
                    name=DATA_PATH_FILES_MOSAIC["WES CNV deletion"],
                    loader=CSVDataLoader(
                        bucket=MOSAIC_BUCKET,
                        data_transformer=MOSAICWESDataCleaner(),
                    ),
                ),
                "WES CNV amplification": DataFile(
                    name=DATA_PATH_FILES_MOSAIC["WES CNV amplification"],
                    loader=CSVDataLoader(
                        bucket=MOSAIC_BUCKET,
                        data_transformer=MOSAICWESDataCleaner(),
                    ),
                ),
                "WES CNV oncogenic": DataFile(
                    name=DATA_PATH_FILES_MOSAIC["WES CNV oncogenic"],
                    loader=CSVDataLoader(
                        bucket=MOSAIC_BUCKET,
                        data_transformer=MOSAICWESDataCleaner(),
                    ),
                ),
                "WES mutations": DataFile(
                    name=DATA_PATH_FILES_MOSAIC["WES mutations"],
                    loader=CSVDataLoader(
                        bucket=MOSAIC_BUCKET,
                        data_transformer=MOSAICWESDataCleaner(),
                    ),
                ),
            }
        ),
        SOURCES.he: DataSource(
            files={
                "HE files": DataFile(
                    name=DATA_PATH_FILES_MOSAIC["HE"],
                    loader=PathsLoader(
                        pattern_files="*.tif",
                        data_transformer=MOSAICHEDataCleaner(),
                    ),
                ),
                # "H1 features": DataFile(
                #     name=DATA_PATH_FILES_MOSAIC["HE_features_H1"],
                #     loader=PathsLoader(
                #         pattern_files="*_eHnE.zarr",
                #         data_transformer=MOSAICHEFeaturesDataCleaner(),
                #     ),
                # ),
            },
        ),
    },
)

BruceDataset = DataCenter(
    name="bruce",
    sources={
        SOURCES.metadata: DataSource(
            files={
                "metadata": DataFile(
                    name=DATA_PATH_FILES_BRUCE["Metadata"],
                    loader=CSVDataLoader(
                        bucket=MOSAIC_BUCKET,
                        data_transformer=BruceLightMetadataDataCleaner(),
                    ),
                ),
            }
        ),
        SOURCES.mibi_images: DataSource(
            files={
                "tumor": DataFile(
                    name=DATA_PATH_FILES_BRUCE["MIBI_tumor_images"],
                    loader=PathsLoader(
                        pattern_files="*/tumor/image_data/*.tiff",
                        data_transformer=BruceMIBIImagesCleaner(),
                    ),
                ),
                "immune": DataFile(
                    name=DATA_PATH_FILES_BRUCE["MIBI_immune_images"],
                    loader=PathsLoader(
                        pattern_files="*/immune/image_data/*.tiff",
                        data_transformer=BruceMIBIImagesCleaner(),
                    ),
                ),
            },
        ),
    },
)
