from typing import Any, Callable, Dict, List, Literal, Union

import narwhals.stable.v1 as nw
from haystack import Document
from haystack.components.converters.utils import normalize_metadata
from narwhals.typing import IntoDataFrame

ReaderFunc = Callable[..., IntoDataFrame]
PandasFileFormat = Literal["csv", "fwf", "json", "html", "xml", "excel", "feather", "parquet", "orc", "pickle"]
PolarsFileFormat = Literal["avro", "csv", "delta", "excel", "ipc", "json", "parquet"]

FileFormat = Union[PandasFileFormat, PolarsFileFormat]


def read_with_select(
    reader_function: ReaderFunc,
    file_path: str,
    columns_subset: Union[List[str], None] = None,
) -> nw.DataFrame:
    df = reader_function(file_path)
    df = nw.from_native(df, eager_only=True)
    if columns_subset:
        df = df.select(columns_subset)
    return df


def frame_to_documents(
    df: nw.DataFrame,
    *,
    content_column: str,
    meta_columns: Union[List[str], None] = None,
    index_column: Union[str, None] = None,
    extra_metadata: Union[Dict[str, Any], List[Dict[str, Any]], None] = None,
) -> List[Document]:
    meta_list = normalize_metadata(extra_metadata, sources_count=df.shape[0])
    documents = []
    for i, row in enumerate(df.iter_rows(named=True)):
        doc_id = str(row.pop(index_column)) if index_column else None
        content = row.pop(content_column)
        meta_row = {k: v for k, v in row.items() if k in meta_columns} if meta_columns else {}
        metadata = {**meta_row, **meta_list[i]} if meta_list else meta_row
        doc = Document(id=doc_id, content=content, meta=metadata)
        documents.append(doc)
    return documents


def get_polars_readers_map() -> Dict[str, ReaderFunc]:  # pragma: no cover
    try:
        import polars as pl
    except ImportError as e:
        msg = "`polars` is not installed. Please run 'pip install \"dataframes-haystack[polars]\"'"
        raise ImportError(msg) from e

    return {
        "avro": pl.read_avro,
        "csv": pl.read_csv,
        "delta": pl.read_delta,
        "excel": pl.read_excel,
        "ipc": pl.read_ipc,
        "json": pl.read_json,
        "parquet": pl.read_parquet,
    }


def get_pandas_readers_map() -> Dict[str, ReaderFunc]:  # pragma: no cover
    import pandas as pd

    return {
        "csv": pd.read_csv,
        "fwf": pd.read_fwf,
        "json": pd.read_json,
        "html": pd.read_html,
        "xml": pd.read_xml,
        "excel": pd.read_excel,
        "feather": pd.read_feather,
        "parquet": pd.read_parquet,
        "orc": pd.read_orc,
        "pickle": pd.read_pickle,
        "sql": pd.read_sql,
        "gbq": pd.read_gbq,
    }
