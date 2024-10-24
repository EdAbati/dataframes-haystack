from functools import partial
from typing import Any, Callable, Dict, List, Optional, Union

import narwhals.stable.v1 as nw
import pandas as pd
from haystack import Document, component, logging

from dataframes_haystack.components.converters._utils import PandasFileFormat as FileFormat
from dataframes_haystack.components.converters._utils import (
    frame_to_documents,
    get_pandas_readers_map,
    read_with_select,
)

logger = logging.getLogger(__name__)


@component
class FileToPandasDataFrame:
    """Converts files to a pandas.DataFrame.

    Usage example:
    ```python
    from dataframes_haystack.components.converters.pandas import FileToPandasDataFrame

    converter = FileToPandasDataFrame()
    results = converter.run(files=["file1.csv", "file2.csv"])
    df = results["dataframe"]
    print(df.head())
    ```
    """

    def __init__(
        self,
        file_format: FileFormat = "csv",
        read_kwargs: Union[Dict[str, Any], None] = None,
        columns_subset: Union[List[str], None] = None,
    ) -> None:
        """Create a FileToPandasDataFrame component.

        Please refer to the pandas documentation for more information on the supported readers and their parameters: https://pandas.pydata.org/docs/user_guide/io.html

        Args:
            file_format: The format of the files to read. Supported formats are "csv", "fwf", "json", "html", "xml",
              "excel", "feather", "parquet", "orc", and "pickle".
            read_kwargs: Optional keyword arguments to pass to the pandas reader function.
            columns_subset: Optional list of column names to select from the DataFrame after reading the file.
        """
        self.file_format = file_format
        self._reader_function = self._get_read_function()
        self.read_kwargs = read_kwargs or {}
        self.columns_subset = columns_subset

    def _get_read_function(self) -> Callable[..., pd.DataFrame]:
        """Returns the function to read files based on the file format."""
        file_format_mapping = get_pandas_readers_map()
        reader_function = file_format_mapping.get(self.file_format)
        if reader_function:
            return reader_function
        msg = f"Unsupported file format: {self.file_format}"
        raise ValueError(msg)

    def _read_with_select(self, file_path: str) -> nw.DataFrame:
        """Reads a file and selects a subset of columns, if provided."""
        read_func = partial(self._reader_function, **self.read_kwargs)
        return read_with_select(read_func, file_path, self.columns_subset)

    @component.output_types(dataframe=pd.DataFrame)
    def run(self, file_paths: List[str]) -> Dict[str, pd.DataFrame]:
        """Converts files to a pandas.DataFrame.

        Args:
            file_paths: List of file paths.

        Returns:
            A dictionary with the following keys:
            - `dataframe`: pandas.DataFrame containing the content of the files.
        """
        df_list = [self._read_with_select(path) for path in file_paths]
        df = nw.concat(df_list, how="vertical")
        pandas_df = nw.to_native(df)
        return {"dataframe": pandas_df}


@component
class PandasDataFrameConverter:
    """Converts data in a pandas.DataFrame to Documents.

    Usage example:
    ```python
    from dataframes_haystack.components.converters.pandas import PandasDataFrameConverter

    converter = PandasDataFrameConverter(content_column="text")
    results = converter.run(dataframe=df)
    documents = results["documents"]
    print(documents[0].content)
    ```
    """

    def __init__(
        self,
        content_column: str,
        meta_columns: Optional[List[str]] = None,
        use_index_as_id: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        """Create a PandasDataFrameConverter component.

        Args:
            content_column: The name of the column in the DataFrame that contains the text content.
            meta_columns: Optional list of column names in the DataFrame that contain metadata.
            use_index_as_id: If True, the index of the DataFrame will be used as the ID of the Documents.
        """
        self.content_column = content_column
        self.meta_columns = meta_columns or []
        self.use_index_as_id = use_index_as_id

    def _is_compatible_index(self, dataframe: pd.DataFrame) -> bool:
        """Returns True if the index of the DataFrame can be used as the ID of the Documents."""
        return not (self.use_index_as_id and isinstance(dataframe.index, pd.MultiIndex))

    @component.output_types(documents=List[Document])
    def run(
        self,
        dataframe: pd.DataFrame,
        meta: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    ) -> Dict[str, List[Document]]:
        """Converts text files to Documents.

        Args:
            dataframe:
                pandas.DataFrame containing text content and metadata.
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
        if not self._is_compatible_index(dataframe):
            msg = (
                "The index of the DataFrame cannot be used as the ID of the Documents."
                "Please make sure that the index is not a MultiIndex or set `use_index_as_id` to False."
            )
            raise ValueError(msg)

        if self.use_index_as_id:
            index_col_name = "__temp_index_col__"
            dataframe = dataframe.assign(**{index_col_name: dataframe.index.astype(str)})
            selected_columns = [index_col_name, self.content_column, *self.meta_columns]
        else:
            index_col_name = None
            selected_columns = [self.content_column, *self.meta_columns]

        df = nw.from_native(dataframe, eager_only=True)
        df = df.select(selected_columns)
        documents = frame_to_documents(
            df,
            content_column=self.content_column,
            meta_columns=self.meta_columns,
            index_column=index_col_name,
            extra_metadata=meta,
        )
        return {"documents": documents}
