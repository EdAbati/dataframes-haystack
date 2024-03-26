from typing import Any, Dict, List, Union

import pandas as pd
import pytest

from dataframes_haystack.components.converters.pandas import PandasDataFrameConverter


@pytest.fixture(scope="function")
def pandas_dataframe():
    return pd.DataFrame(
        data={
            "content": ["content1", "content2"],
            "meta1": ["meta1_1", "meta1_2"],
            "meta2": ["meta2_1", "meta2_2"],
        },
        index=[0, 1],
    )


def test_pandas_dataframe_default_converter(pandas_dataframe: pd.DataFrame):
    converter = PandasDataFrameConverter(content_column="content")
    results = converter.run(dataframe=pandas_dataframe)
    documents = results["documents"]
    assert len(documents) == 2
    assert [doc.id for doc in documents] != ["0", "1"]
    assert [doc.content for doc in documents] == ["content1", "content2"]
    assert [doc.meta for doc in documents] == [{}, {}]
    assert [doc.embedding for doc in documents] == [None, None]


@pytest.mark.parametrize("use_index_as_id", [True, False])
def test_pandas_dataframe_converter_use_index_as_id(pandas_dataframe: pd.DataFrame, use_index_as_id: bool):
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
    "meta_columns, expected_meta",
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
):
    converter = PandasDataFrameConverter(content_column="content", meta_columns=meta_columns)
    results = converter.run(dataframe=pandas_dataframe)
    documents = results["documents"]
    assert [doc.id for doc in documents] != ["0", "1"]
    assert [doc.content for doc in documents] == ["content1", "content2"]
    assert [doc.meta for doc in documents] == expected_meta


@pytest.mark.parametrize(
    "meta, expected_meta",
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
):
    converter = PandasDataFrameConverter(content_column="content", meta_columns=["meta1"])
    results = converter.run(dataframe=pandas_dataframe, meta=meta)
    documents = results["documents"]
    assert [doc.id for doc in documents] != ["0", "1"]
    assert [doc.content for doc in documents] == ["content1", "content2"]
    assert [doc.meta for doc in documents] == expected_meta


def test_pandas_dataframe_converters_multindex_error(
    pandas_dataframe: pd.DataFrame,
):
    pandas_dataframe.index = pd.MultiIndex.from_tuples([("a", 0), ("b", 1)])
    converter = PandasDataFrameConverter(content_column="content", use_index_as_id=True)
    with pytest.raises(ValueError):
        converter.run(dataframe=pandas_dataframe)


def test_converter_in_pipeline():
    from textwrap import dedent

    from haystack.components.preprocessors import DocumentCleaner
    from haystack.core.pipeline import Pipeline

    pipeline = Pipeline()
    pipeline.add_component("converter", PandasDataFrameConverter(content_column="content"))
    pipeline.add_component("cleaner", DocumentCleaner())
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
    assert dedent(converter_expected_yaml) in yaml_pipeline

    new_pipeline = Pipeline.loads(yaml_pipeline)
    assert yaml_pipeline == new_pipeline.dumps()
