__doc__ = "This scripts loads data in `.npz` and `.parquet` format into public Kaggle repository"

import kagglehub
from kagglehub.config import get_kaggle_credentials

from pathlib import Path


username = get_kaggle_credentials().username

DATASET_HANDLE = f'{username}/City-Traffic-Benchmarks'
LOCAL_DATA_DIR = str(Path(__file__).parent.parent / "data" / "traffic") # <<--- we stored files there for upload
print(LOCAL_DATA_DIR)

kagglehub.dataset_upload(DATASET_HANDLE, LOCAL_DATA_DIR)
