from pathlib import Path
from typing import Any, Dict, List, Union

import narwhals as nw
import pandas as pd
import polars as pl
import pytest
from haystack import Document
from narwhals.typing import IntoDataFrame

from dataframes_haystack.components.converters._utils import ReaderFunc, frame_to_documents, read_with_select


@pytest.mark.parametrize("reader_func", [pd.read_csv, pl.read_csv])
def test_read_with_select(reader_func: ReaderFunc, csv_file_path: Path) -> None:
    df = read_with_select(reader_func, str(csv_file_path))
    assert df.shape == (2, 3)
    output = {col: series.to_list() for col, series in df.to_dict().items()}
    expected = {
        "content": ["content1", "content2"],
        "meta1": ["meta1_1", "meta1_2"],
        "meta2": ["meta2_1", "meta2_2"],
    }
    assert output == expected


@pytest.mark.parametrize("reader_func", [pd.read_csv, pl.read_csv])
def test_read_with_select_subset(reader_func: ReaderFunc, csv_file_path: Path) -> None:
    df = read_with_select(reader_func, str(csv_file_path), columns_subset=["content", "meta2"])
    assert df.shape == (2, 2)
    output = {col: series.to_list() for col, series in df.to_dict().items()}
    expected = {
        "content": ["content1", "content2"],
        "meta2": ["meta2_1", "meta2_2"],
    }
    assert output == expected


@pytest.mark.parametrize(
    ("meta_columns", "index_column", "extra_metadata", "expected_docs"),
    [
        (
            ["meta1", "meta2"],
            None,
            None,
            [
                Document(content="content1", meta={"meta1": "meta1_1", "meta2": "meta2_1"}),
                Document(content="content2", meta={"meta1": "meta1_2", "meta2": "meta2_2"}),
            ],
        ),
        (
            ["meta2"],
            "meta1",
            None,
            [
                Document(id="meta1_1", content="content1", meta={"meta2": "meta2_1"}),
                Document(id="meta1_2", content="content2", meta={"meta2": "meta2_2"}),
            ],
        ),
        (
            ["meta1"],
            None,
            {"extra": "metadata"},
            [
                Document(content="content1", meta={"meta1": "meta1_1", "extra": "metadata"}),
                Document(content="content2", meta={"meta1": "meta1_2", "extra": "metadata"}),
            ],
        ),
        (
            None,
            None,
            [{"extra": "metadata1"}, {"extra": "metadata2"}],
            [
                Document(content="content1", meta={"extra": "metadata1"}),
                Document(content="content2", meta={"extra": "metadata2"}),
            ],
        ),
    ],
)
def test_frame_to_documents(
    dataframe: IntoDataFrame,
    meta_columns: Union[List[str], None],
    index_column: Union[str, None],
    extra_metadata: Union[Dict[str, Any], List[Dict[str, Any]], None],
    expected_docs: List[Document],
) -> None:
    documents = frame_to_documents(
        nw.from_native(dataframe),
        content_column="content",
        meta_columns=meta_columns,
        index_column=index_column,
        extra_metadata=extra_metadata,
    )
    assert len(documents) == 2
    for doc, expected_doc in zip(documents, expected_docs):
        assert doc.content == expected_doc.content
        assert doc.meta == expected_doc.meta
