from pathlib import Path

import pandas as pd
import polars as pl
import pytest
from narwhals.typing import IntoDataFrame

DATA = {
    "content": ["content1", "content2"],
    "meta1": ["meta1_1", "meta1_2"],
    "meta2": ["meta2_1", "meta2_2"],
}

CSV_STRING = """content,meta1,meta2
content1,meta1_1,meta2_1
content2,meta1_2,meta2_2
"""


def _pandas_df() -> pd.DataFrame:
    return pd.DataFrame(data=DATA, index=[0, 1])


def _polars_df() -> pl.DataFrame:
    return pl.DataFrame(data=DATA)


@pytest.fixture
def pandas_dataframe() -> pd.DataFrame:
    return _pandas_df()


@pytest.fixture
def polars_dataframe() -> pl.DataFrame:
    return _polars_df()


@pytest.fixture(params=[_pandas_df, _polars_df])
def dataframe(request: pytest.FixtureRequest) -> IntoDataFrame:
    return request.param()


@pytest.fixture(scope="session")
def csv_file_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp_path = tmp_path_factory.mktemp("data") / "data.csv"
    tmp_path.write_text(CSV_STRING)
    return tmp_path
