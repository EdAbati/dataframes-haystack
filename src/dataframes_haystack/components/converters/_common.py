from functools import partial
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import narwhals.stable.v1 as nw
from haystack import Document, component, logging
from haystack.components.converters.utils import normalize_metadata
from narwhals.typing import IntoDataFrame

logger = logging.getLogger(__name__)

PandasFileFormat = Literal["csv", "fwf", "json", "html", "xml", "excel", "feather", "parquet", "orc", "pickle"]
PolarsFileFormat = Literal["avro", "csv", "delta", "excel", "ipc", "json", "parquet"]

FileFormat = Union[PandasFileFormat, PolarsFileFormat]

Backends = Literal["pandas", "polars"]

ReaderFunc = Callable[..., IntoDataFrame]


def read_with_select(
    reader_function: ReaderFunc, file_path: str, columns_subset: Union[List[str], None] = None
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


@component
class DataFrameFileToDocument:
    """
    Reads files and converts their data in Documents.

    Usage example:
    ```python
    from dataframes_haystack.components.converters import DataFrameFileToDocument

    converter = DataFrameFileToDocument()
    results = converter.run(files=["file1.csv", "file2.csv"])
    documents = results["documents"]
    print(documents[0].content)
    ```
    """

    def __init__(
        self,
        content_column: str,
        meta_columns: Union[List[str], None] = None,
        index_column: Union[str, None] = None,
        file_format: FileFormat = "csv",
        read_kwargs: Optional[Dict[str, Any]] = None,
        backend: Backends = "polars",
    ):
        """
        Create a DataFrameFileToDocument component.

        Args:
            content_column: The name of the column in the DataFrame that contains the text content.
            meta_columns: Optional list of column names in the DataFrame that contain metadata.
            index_column: The name of the column in the DataFrame that contains the index.
            file_format: The format of the files to read.
            read_kwargs: Optional keyword arguments to pass to the file reader function.
            backend: The backend to use for reading the files.
        """
        self.content_column = content_column
        self.meta_columns = meta_columns or []
        self.index_column = index_column
        self.file_format = file_format
        self.read_kwargs = read_kwargs or {}
        self.backend = backend
        if self.backend not in ["pandas", "polars"]:
            msg = f"Unsupported backend: {self.backend}"
            raise ValueError(msg)
        self._reader_function = self._get_reader_function()

    def _get_reader_function(self):
        if self.backend == "pandas":
            return self._get_reader_function_pandas()
        elif self.backend == "polars":
            return self._get_reader_function_polars()

    def _get_reader_function_polars(self):
        try:
            import polars as pl
        except ImportError as e:
            msg = "`polars` is not installed. Please run 'pip install \"dataframes-haystack[polars]\"'"
            raise ImportError(msg) from e

        file_format_mapping = {
            "avro": pl.read_avro,
            "csv": pl.read_csv,
            "delta": pl.read_delta,
            "excel": pl.read_excel,
            "ipc": pl.read_ipc,
            "json": pl.read_json,
            "parquet": pl.read_parquet,
        }
        reader_function = file_format_mapping.get(self.file_format)
        if reader_function:
            return reader_function
        msg = f"Unsupported file format in polars: {self.file_format}"
        raise ValueError(msg)

    def _get_reader_function_pandas(self):
        msg = "Pandas reader function not implemented yet"
        raise NotImplementedError(msg)

    def _run_read(self, file_paths: List[str]) -> nw.DataFrame:
        selected_columns = [self.index_column, self.content_column, *self.meta_columns]
        read_func = partial(self._reader_function, **self.read_kwargs)
        df_list = [read_with_select(read_func, file_path=path, columns_subset=selected_columns) for path in file_paths]
        return nw.concat(df_list, how="vertical")

    @component.output_types(documents=List[Document])
    def run(self, file_paths: List[str], meta: Union[Dict[str, Any], List[Dict[str, Any]], None] = None):
        """
        Reads files and converts their data in Documents.

        Args:
            file_paths: List of file paths to read.
            meta:
                Optional metadata to attach to the Documents.
                This value can be either a dictionary or a list of dictionaries.
                If it's a dictionary, its content is added to the metadata of all produced Documents.
                If it's a list, the length of the list must match the number of rows in the DataFrame,
                because the two lists will be zipped.

        Returns:
            A dictionary with the following keys:
            - `documents`: Created Documents
        """
        documents = []
        df = self._run_read(file_paths)
        meta_list = normalize_metadata(meta, sources_count=df.shape[0])
        for i, row in enumerate(df.iter_rows(named=True)):
            doc_id = str(row.pop(self.index_column)) if self.index_column else None
            content = row.pop(self.content_column)
            meta_row = {k: v for k, v in row.items() if k in self.meta_columns} if self.meta_columns else {}
            metadata = {**meta_row, **meta_list[i]} if meta_list else meta_row
            doc = Document(id=doc_id, content=content, meta=metadata)
            documents.append(doc)
        return {"documents": documents}
