from typing import Any, Dict, List, Union

import polars as pl
import pytest

from dataframes_haystack.components.converters.polars import PolarsDataFrameConverter


@pytest.fixture(scope="function")
def polars_dataframe():
    return pl.DataFrame(
        data={
            "content": ["content1", "content2"],
            "meta1": ["meta1_1", "meta1_2"],
            "meta2": ["meta2_1", "meta2_2"],
        },
    )


def test_polars_dataframe_default_converter(polars_dataframe: pl.DataFrame):
    converter = PolarsDataFrameConverter(content_column="content")
    results = converter.run(dataframe=polars_dataframe)
    documents = results["documents"]
    assert len(documents) == 2
    assert [doc.id for doc in documents] != ["0", "1"]
    assert [doc.content for doc in documents] == ["content1", "content2"]
    assert [doc.meta for doc in documents] == [{}, {}]
    assert [doc.embedding for doc in documents] == [None, None]


def test_polars_dataframe_converter_index_column(polars_dataframe: pl.DataFrame):
    polars_dataframe = polars_dataframe.with_columns(pl.Series("index", [0, 1]))
    converter = PolarsDataFrameConverter(content_column="content", index_column="index")
    results = converter.run(dataframe=polars_dataframe)
    documents = results["documents"]
    assert [doc.id for doc in documents] == ["0", "1"]
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
def test_polars_dataframe_converter_meta_columns(
    polars_dataframe: pl.DataFrame,
    meta_columns: List[str],
    expected_meta: List[Dict[str, str]],
):
    converter = PolarsDataFrameConverter(content_column="content", meta_columns=meta_columns)
    results = converter.run(dataframe=polars_dataframe)
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
                {
                    "meta1": "meta1_1",
                    "extra_meta_1": "value1",
                    "extra_meta_2": "value2",
                },
                {
                    "meta1": "meta1_2",
                    "extra_meta_1": "value1",
                    "extra_meta_2": "value2",
                },
            ],
        ),
    ],
)
def test_polars_dataframe_converter_all_metadata(
    polars_dataframe: pl.DataFrame,
    meta: Union[Dict[str, Any], List[Dict[str, Any]]],
    expected_meta: List[Dict[str, str]],
):
    converter = PolarsDataFrameConverter(content_column="content", meta_columns=["meta1"])
    results = converter.run(dataframe=polars_dataframe, meta=meta)
    documents = results["documents"]
    assert [doc.id for doc in documents] != ["0", "1"]
    assert [doc.content for doc in documents] == ["content1", "content2"]
    assert [doc.meta for doc in documents] == expected_meta


def test_converter_in_pipeline():
    from textwrap import dedent

    from haystack.components.preprocessors import DocumentCleaner
    from haystack.core.pipeline import Pipeline

    pipeline = Pipeline()
    pipeline.add_component("converter", PolarsDataFrameConverter(content_column="content"))
    pipeline.add_component("cleaner", DocumentCleaner())
    pipeline.connect("converter", "cleaner")

    yaml_pipeline = pipeline.dumps()

    converter_expected_yaml = """\
      converter:
          init_parameters:
            content_column: content
            index_column: null
            meta_columns: []
          type: dataframes_haystack.components.converters.polars.PolarsDataFrameConverter
    """
    assert dedent(converter_expected_yaml) in yaml_pipeline

    new_pipeline = Pipeline.loads(yaml_pipeline)
    assert yaml_pipeline == new_pipeline.dumps()
