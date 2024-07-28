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

    def _run_read(self, files: List[str]) -> nw.DataFrame:
        selected_columns = [self.index_column, self.content_column, *self.meta_columns]
        df_list = [self._reader_function(file, **self.read_kwargs).select(selected_columns) for file in files]
        return nw.concat(df_list, how="vertical")

    @component.output_types(documents=List[Document])
    def run(self, files: List[str]):
        """
        Reads files and converts their data in Documents.

        Args:
            files: List of file paths to read.

        Returns:
            Dictionary containing the list of Documents.
        """
        documents = []
        selected_columns = [self.index_column, self.content_column, *self.meta_columns]
        _ = selected_columns
        df = self._run_read(files)
        _ = df.iter_rows(named=True)
        # TODO here
        for file in files:
            df = self._reader_function(file, **self.read_kwargs)
            for _, row in df.iterrows():
                content = row[self.content_column]
                metadata = normalize_metadata(row[self.meta_columns])
                if self.index_column:
                    id_ = row[self.index_column]
                else:
                    id_ = None
                documents.append(Document(content=content, id=id_, meta=metadata))
        return {"documents": documents}
