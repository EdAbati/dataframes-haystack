from pathlib import Path

import pandas as pd
import polars as pl
import pytest

DATA = {
    "content": ["content1", "content2"],
    "meta1": ["meta1_1", "meta1_2"],
    "meta2": ["meta2_1", "meta2_2"],
}

CSV_STRING = """content,meta1,meta2
content1,meta1_1,meta2_1
content2,meta1_2,meta2_2
"""


@pytest.fixture(scope="function")
def pandas_dataframe():
    return pd.DataFrame(data=DATA, index=[0, 1])


@pytest.fixture(scope="function")
def polars_dataframe():
    return pl.DataFrame(data=DATA)


@pytest.fixture(scope="session")
def csv_file_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp_path = tmp_path_factory.mktemp("data") / "data.csv"
    tmp_path.write_text(CSV_STRING)
    return tmp_path
