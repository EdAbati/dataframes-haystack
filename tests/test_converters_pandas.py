from pathlib import Path
from typing import Any, Dict, List, Union

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from dataframes_haystack.components.converters.pandas import FileToPandasDataFrame, PandasDataFrameConverter
from tests.utils import assert_pipeline_yaml_equal


def test_pandas_dataframe_default_converter(pandas_dataframe: pd.DataFrame) -> None:
    converter = PandasDataFrameConverter(content_column="content")
    results = converter.run(dataframe=pandas_dataframe)
    documents = results["documents"]
    assert len(documents) == 2
    assert [doc.id for doc in documents] != ["0", "1"]
    assert [doc.content for doc in documents] == ["content1", "content2"]
    assert [doc.meta for doc in documents] == [{}, {}]
    assert [doc.embedding for doc in documents] == [None, None]


@pytest.mark.parametrize("use_index_as_id", [True, False])
def test_pandas_dataframe_converter_use_index_as_id(
    pandas_dataframe: pd.DataFrame,
    use_index_as_id: bool,  # noqa: FBT001
) -> None:
    converter = PandasDataFrameConverter(content_column="content", use_index_as_id=use_index_as_id)
    results = converter.run(dataframe=pandas_dataframe)
    documents = results["documents"]
    ids = [doc.id for doc in documents]
    if use_index_as_id:
        assert ids == ["0", "1"]
    else:
        assert ids != ["0", "1"]
    assert [doc.content for doc in documents] == ["content1", "content2"]
    assert [doc.meta for doc in documents] == [{}, {}]


@pytest.mark.parametrize(
    ("meta_columns", "expected_meta"),
    [
        (["meta1"], [{"meta1": "meta1_1"}, {"meta1": "meta1_2"}]),
        (["meta2"], [{"meta2": "meta2_1"}, {"meta2": "meta2_2"}]),
        (
            ["meta1", "meta2"],
            [
                {"meta1": "meta1_1", "meta2": "meta2_1"},
                {"meta1": "meta1_2", "meta2": "meta2_2"},
            ],
        ),
    ],
)
def test_pandas_dataframe_converter_meta_columns(
    pandas_dataframe: pd.DataFrame,
    meta_columns: List[str],
    expected_meta: List[Dict[str, str]],
) -> None:
    converter = PandasDataFrameConverter(content_column="content", meta_columns=meta_columns)
    results = converter.run(dataframe=pandas_dataframe)
    documents = results["documents"]
    assert [doc.id for doc in documents] != ["0", "1"]
    assert [doc.content for doc in documents] == ["content1", "content2"]
    assert [doc.meta for doc in documents] == expected_meta


@pytest.mark.parametrize(
    ("meta", "expected_meta"),
    [
        (
            [{"extra_meta_1": "value1"}, {"extra_meta_2": "value2"}],
            [
                {"meta1": "meta1_1", "extra_meta_1": "value1"},
                {"meta1": "meta1_2", "extra_meta_2": "value2"},
            ],
        ),
        (
            {"extra_meta_1": "value1", "extra_meta_2": "value2"},
            [
                {"meta1": "meta1_1", "extra_meta_1": "value1", "extra_meta_2": "value2"},
                {"meta1": "meta1_2", "extra_meta_1": "value1", "extra_meta_2": "value2"},
            ],
        ),
    ],
)
def test_pandas_dataframe_converter_all_metadata(
    pandas_dataframe: pd.DataFrame,
    meta: Union[Dict[str, Any], List[Dict[str, Any]]],
    expected_meta: List[Dict[str, str]],
) -> None:
    converter = PandasDataFrameConverter(content_column="content", meta_columns=["meta1"])
    results = converter.run(dataframe=pandas_dataframe, meta=meta)
    documents = results["documents"]
    assert [doc.id for doc in documents] != ["0", "1"]
    assert [doc.content for doc in documents] == ["content1", "content2"]
    assert [doc.meta for doc in documents] == expected_meta


def test_pandas_dataframe_converters_multindex_error(
    pandas_dataframe: pd.DataFrame,
) -> None:
    pandas_dataframe.index = pd.MultiIndex.from_tuples([("a", 0), ("b", 1)])
    converter = PandasDataFrameConverter(content_column="content", use_index_as_id=True)
    with pytest.raises(ValueError, match="The index of the DataFrame cannot be used"):
        converter.run(dataframe=pandas_dataframe)


@pytest.mark.parametrize("column_subset", [None, ["content"], ["content", "meta1"]])
def test_file_to_pandas_converter(
    csv_file_path: Path,
    pandas_dataframe: pd.DataFrame,
    column_subset: Union[List[str], None],
) -> None:
    converter = FileToPandasDataFrame(columns_subset=column_subset)
    results = converter.run(file_paths=[str(csv_file_path)])
    if column_subset:
        pandas_dataframe = pandas_dataframe[column_subset]
    assert_frame_equal(results["dataframe"], pandas_dataframe)


def test_file_to_pandas_converter_read_kwargs(csv_file_path: Path, pandas_dataframe: pd.DataFrame) -> None:
    cols_to_select = ["content", "meta2"]
    converter = FileToPandasDataFrame(read_kwargs={"usecols": cols_to_select})
    results = converter.run(file_paths=[str(csv_file_path)])
    assert_frame_equal(results["dataframe"], pandas_dataframe[cols_to_select])


def test_file_to_pandas_converter_valueerror() -> None:
    with pytest.raises(ValueError, match="Unsupported file format"):
        FileToPandasDataFrame(file_format="foo")


def test_converter_in_pipeline() -> None:
    from textwrap import dedent

    from haystack.components.preprocessors import DocumentCleaner
    from haystack.core.pipeline import Pipeline

    pipeline = Pipeline()
    pipeline.add_component("file_to_pandas", FileToPandasDataFrame())
    pipeline.add_component("converter", PandasDataFrameConverter(content_column="content"))
    pipeline.add_component("cleaner", DocumentCleaner())
    pipeline.connect("file_to_pandas", "converter")
    pipeline.connect("converter", "cleaner")

    yaml_pipeline = pipeline.dumps()

    converter_expected_yaml = """\
      converter:
          init_parameters:
            content_column: content
            meta_columns: []
            use_index_as_id: false
          type: dataframes_haystack.components.converters.pandas.PandasDataFrameConverter
    """
    file_to_pandas_expected_yaml = """\
      file_to_pandas:
          init_parameters:
            columns_subset: null
            file_format: csv
            read_kwargs: {}
          type: dataframes_haystack.components.converters.pandas.FileToPandasDataFrame
    """
    assert dedent(converter_expected_yaml) in yaml_pipeline
    assert dedent(file_to_pandas_expected_yaml) in yaml_pipeline

    new_pipeline = Pipeline.loads(yaml_pipeline)
    assert_pipeline_yaml_equal(yaml_pipeline, new_pipeline.dumps())
