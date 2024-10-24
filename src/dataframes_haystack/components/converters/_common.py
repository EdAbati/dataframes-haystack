import logging
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Union

import narwhals.stable.v1 as nw
from haystack import Document, component

from dataframes_haystack.components.converters._utils import (
    FileFormat,
    ReaderFunc,
    frame_to_documents,
    get_pandas_readers_map,
    get_polars_readers_map,
    read_with_select,
)

logger = logging.getLogger(__name__)

Backends = Literal["pandas", "polars"]


@component
class DataFrameFileToDocument:
    """Reads files and converts their data in Documents.

    Usage example:
    ```python
    from dataframes_haystack.components.converters import DataFrameFileToDocument

    converter = DataFrameFileToDocument(content_column="text_str")
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
    ) -> None:
        """Create a DataFrameFileToDocument component.

        Args:
            content_column: The name of the DataFrame column that contains the text content.
            meta_columns: Optional list of names of the DataFrame columns that contain metadata.
            index_column: The name of the DataFrame column that contains the index.
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

    def _get_reader_function(self) -> ReaderFunc:
        file_format_mapping = get_pandas_readers_map() if self.backend == "pandas" else get_polars_readers_map()
        reader_function = file_format_mapping.get(self.file_format)
        if reader_function:
            return reader_function
        msg = f"Unsupported file format for {self.backend} backend: {self.file_format}"
        raise ValueError(msg)

    def _run_read(self, file_paths: List[str]) -> nw.DataFrame:
        selected_columns = [self.index_column, self.content_column, *self.meta_columns]
        selected_columns = [col for col in selected_columns if col is not None]
        read_func = partial(self._reader_function, **self.read_kwargs)
        df_list = [read_with_select(read_func, file_path=path, columns_subset=selected_columns) for path in file_paths]
        return nw.concat(df_list, how="vertical")

    @component.output_types(documents=List[Document])
    def run(
        self,
        file_paths: List[str],
        meta: Union[Dict[str, Any], List[Dict[str, Any]], None] = None,
    ) -> Dict[str, List[Document]]:
        """Reads files and converts their data in Documents.

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
        df = self._run_read(file_paths)
        documents = frame_to_documents(
            df,
            content_column=self.content_column,
            meta_columns=self.meta_columns,
            index_column=self.index_column,
            extra_metadata=meta,
        )

        return {"documents": documents}
