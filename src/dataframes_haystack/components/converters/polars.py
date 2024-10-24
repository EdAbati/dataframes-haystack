from functools import partial
from typing import Any, Callable, Dict, List, Optional, Union

import narwhals.stable.v1 as nw
from haystack import Document, component, logging

from dataframes_haystack.components.converters._utils import PolarsFileFormat as FileFormat
from dataframes_haystack.components.converters._utils import (
    frame_to_documents,
    get_polars_readers_map,
    read_with_select,
)

try:
    import polars as pl
except ImportError as e:
    msg = "`polars` is not installed. Please run 'pip install \"dataframes-haystack[polars]\"'"
    raise ImportError(msg) from e

logger = logging.getLogger(__name__)


@component
class FileToPolarsDataFrame:
    """Converts files to a polars.DataFrame.

    Usage example:
    ```python
    from dataframes_haystack.components.converters.polars import FileToPolarsDataFrame

    converter = FileToPolarsDataFrame()
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
    ) -> None:
        """Create a FileToPolarsDataFrame component.

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

    def _get_read_function(self) -> Callable[..., pl.DataFrame]:
        """Returns the function to read files based on the file format."""
        file_format_mapping = get_polars_readers_map()
        reader_function = file_format_mapping.get(self.file_format)
        if reader_function:
            return reader_function
        msg = f"Unsupported file format: {self.file_format}"
        raise ValueError(msg)

    def _read_with_select(self, file_path: str) -> nw.DataFrame:
        """Reads a file and selects only the specified columns."""
        read_func = partial(self._reader_function, **self.read_kwargs)
        return read_with_select(read_func, file_path, self.columns_subset)

    @component.output_types(dataframe=pl.DataFrame)
    def run(self, file_paths: List[str]) -> Dict[str, pl.DataFrame]:
        """Converts files to a polars.DataFrame.

        Args:
            file_paths: List of file paths to read.

        Returns:
            A dictionary with the following keys:
            - `dataframe`: The polars.DataFrame created from the files.
        """
        df_list = [self._read_with_select(path) for path in file_paths]
        df = nw.concat(df_list, how="vertical")
        polars_df = nw.to_native(df)
        return {"dataframe": polars_df}


@component
class PolarsDataFrameConverter:
    """Converts data in a polars.DataFrame to Documents.

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
    ) -> None:
        """Create a PolarsDataFrameConverter component.

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
    ) -> Dict[str, List[Document]]:
        """Converts data in a polars.DataFrame to Documents.

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
        df = nw.from_native(dataframe)
        selected_columns = [self.index_column, self.content_column, *self.meta_columns]
        selected_columns = [col for col in selected_columns if col is not None]
        df = df.select(selected_columns)
        documents = frame_to_documents(
            df,
            content_column=self.content_column,
            meta_columns=self.meta_columns,
            index_column=self.index_column,
            extra_metadata=meta,
        )
        return {"documents": documents}
