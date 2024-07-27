from typing import Any, Dict, List, Literal, Optional, Union

from haystack import Document, component, logging
from haystack.components.converters.utils import normalize_metadata

try:
    import polars as pl
except ImportError as e:
    msg = "`polars` is not installed. Please run 'pip install \"dataframes-haystack[polars]\"'"
    raise ImportError(msg) from e

logger = logging.getLogger(__name__)


FileFormat = Literal["avro", "csv", "delta", "excel", "ipc", "json", "parquet"]


@component
class FileToPolarsConverter:
    """
    Converts files to a polars.DataFrame.

    Usage example:
    ```python
    from dataframes_haystack.components.converters.polars import FileToPolarsConverter

    converter = FileToPolarsConverter()
    results = converter.run(files=["file1.csv", "file2.csv"])
    df = results["dataframe"]
    print(df.head())
    ```
    """

    def __init__(
        self,
        file_format: FileFormat = "csv",
        read_kwargs: Optional[Dict[str, Any]] = None,
        columns_subset: Union[List[str], None] = None,
    ):
        """
        Create a FileToPolarsConverter component.

        Please refer to the polars documentation for more information on the supported readers and their parameters: https://docs.pola.rs/api/python/stable/reference/io.html

        Args:
            file_format: The format of the files to read. Supported formats are "avro", "csv", "delta", "excel", "ipc",
              "json", and "parquet".
            read_kwargs: Optional keyword arguments to pass to the polars reader function.
            columns_subset: Optional list of column names to select from the DataFrame after reading the file.
        """
        self.file_format = file_format
        self._reader_function = self._get_read_function()
        self.read_kwargs = read_kwargs or {}
        self.columns_subset = columns_subset

    def _get_read_function(self):
        """Returns the function to read files based on the file format."""

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
        msg = f"Unsupported file format: {self.file_format}"
        raise ValueError(msg)

    def _read_with_select(self, file: str) -> pl.DataFrame:
        """Reads a file and selects only the specified columns."""
        df = self._reader_function(file, **self.read_kwargs)
        if self.columns_subset:
            return df.select(self.columns_subset)
        return df

    @component.output_types(dataframe=pl.DataFrame)
    def run(self, files: List[str]):
        """
        Converts files to a polars.DataFrame.

        Args:
            files: List of file paths to read.

        Returns:
            A dictionary with the following keys:
            - `dataframe`: The polars.DataFrame created from the files.
        """
        df_list = [self._read_with_select(file) for file in files]
        df = pl.concat(df_list, how="vertical")
        return {"dataframe": df}


@component
class PolarsDataFrameConverter:
    """
    Converts data in a polars.DataFrame to Documents.

    Usage example:
    ```python
    from dataframes_haystack.components.converters.polars import PolarsDataFrameConverter

    converter = PolarsDataFrameConverter(content_column="text")
    results = converter.run(dataframe=df)
    documents = results["documents"]
    print(documents[0].content)
    ```
    """

    def __init__(
        self,
        content_column: str,
        meta_columns: Union[List[str], None] = None,
        index_column: Union[str, None] = None,
    ):
        """
        Create a PolarsDataFrameConverter component.

        Args:
            content_column: The name of the column in the DataFrame that contains the text content.
            meta_columns: Optional list of column names in the DataFrame that contain metadata.
            index_column: The name of the column in the DataFrame that contains the index.
        """
        self.content_column = content_column
        self.meta_columns = meta_columns or []
        self.index_column = index_column

    @component.output_types(documents=List[Document])
    def run(
        self,
        dataframe: pl.DataFrame,
        meta: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    ):
        """
        Converts data in a polars.DataFrame to Documents.

        Args:
            dataframe:
                polars.DataFrame containing text content and metadata.
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
        meta_list = normalize_metadata(meta, sources_count=dataframe.shape[0])

        selected_columns = [self.index_column, self.content_column, *self.meta_columns]
        data_rows = dataframe.select(selected_columns).to_dicts()

        documents = []
        for i, row in enumerate(data_rows):
            doc_id = str(row.pop(self.index_column)) if self.index_column else None
            content = row.pop(self.content_column)
            meta_row = {k: v for k, v in row.items() if k in self.meta_columns} if self.meta_columns else {}
            metadata = {**meta_row, **meta_list[i]} if meta_list else meta_row
            doc = Document(id=doc_id, content=content, meta=metadata)
            documents.append(doc)

        return {"documents": documents}
