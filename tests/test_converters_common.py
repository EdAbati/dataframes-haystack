from pathlib import Path
from typing import Any, Dict, List, Union

import pytest
from haystack import Document

from dataframes_haystack.components.converters import DataFrameFileToDocument
from tests.utils import assert_pipeline_yaml_equal


@pytest.mark.parametrize("backend", ["pandas", "polars"])
@pytest.mark.parametrize(
    ("meta_columns", "index_column", "read_kwargs", "expected_docs"),
    [
        (
            None,
            None,
            None,
            [
                Document(content="content1", meta={}),
                Document(content="content2", meta={}),
            ],
        ),
        (
            ["meta2"],
            None,
            None,
            [
                Document(content="content1", meta={"meta2": "meta2_1"}),
                Document(content="content2", meta={"meta2": "meta2_2"}),
            ],
        ),
        (
            ["meta2", "meta1"],
            None,
            None,
            [
                Document(content="content1", meta={"meta2": "meta2_1", "meta1": "meta1_1"}),
                Document(content="content2", meta={"meta2": "meta2_2", "meta1": "meta1_2"}),
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
        (["meta2"], None, {"n_rows": 1}, [Document(content="content1", meta={"meta2": "meta2_1"})]),
    ],
)
def test_dataframe_file_to_document(
    csv_file_path: Path,
    meta_columns: Union[List[str], None],
    index_column: Union[str, None],
    read_kwargs: Union[Dict[str, Any], None],
    expected_docs: List[Document],
    backend: str,
) -> None:
    if backend == "pandas" and read_kwargs and read_kwargs.get("n_rows"):
        read_kwargs = {"nrows": read_kwargs["n_rows"]}
    converter = DataFrameFileToDocument(
        content_column="content",
        meta_columns=meta_columns,
        index_column=index_column,
        read_kwargs=read_kwargs,
        backend=backend,
    )
    results = converter.run(file_paths=[str(csv_file_path)])
    documents = results["documents"]
    for output_doc, expected_doc in zip(documents, expected_docs):
        assert output_doc.content == expected_doc.content
        assert output_doc.meta == expected_doc.meta
        if index_column:
            assert output_doc.id == expected_doc.id


@pytest.mark.parametrize(
    ("kwargs", "error_msg"),
    [
        ({"backend": "foo"}, "Unsupported backend: foo"),
        ({"backend": "pandas", "file_format": "foo"}, "Unsupported file format for pandas backend: foo"),
        ({"backend": "polars", "file_format": "foo"}, "Unsupported file format for polars backend: foo"),
    ],
)
def test_dataframe_file_to_document_valueerror(kwargs: Dict[str, Any], error_msg: str) -> None:
    with pytest.raises(ValueError, match=error_msg):
        DataFrameFileToDocument(content_column="a", **kwargs)


def test_converter_in_pipeline() -> None:
    from textwrap import dedent

    from haystack.components.preprocessors import DocumentCleaner
    from haystack.core.pipeline import Pipeline

    pipeline = Pipeline()
    pipeline.add_component("converter", DataFrameFileToDocument(content_column="content"))
    pipeline.add_component("cleaner", DocumentCleaner())
    pipeline.connect("converter", "cleaner")

    yaml_pipeline = pipeline.dumps()

    converter_expected_yaml = """\
      converter:
          init_parameters:
            backend: polars
            content_column: content
            file_format: csv
            index_column: null
            meta_columns: []
            read_kwargs: {}
          type: dataframes_haystack.components.converters._common.DataFrameFileToDocument
    """
    assert dedent(converter_expected_yaml) in yaml_pipeline

    new_pipeline = Pipeline.loads(yaml_pipeline)
    assert_pipeline_yaml_equal(yaml_pipeline, new_pipeline.dumps())
