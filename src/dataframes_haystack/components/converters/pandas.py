from typing import Any, Dict, List, Optional, Union

import pandas as pd
from haystack import Document, component, logging
from haystack.components.converters.utils import normalize_metadata

logger = logging.getLogger(__name__)


@component
class PandasDataFrameConverter:
    """
    Converts data in a pandas.DataFrame to Documents.

    Usage example:
    ```python
    from dataframes_haystack.components.converters.pandas import PandasDataFrameConverter

    converter = PandasDataFrameToDocument(content_column="text")
    results = converter.run(dataframe=df)
    documents = results["documents"]
    print(documents[0].content)
    ```
    """

    def __init__(
        self,
        content_column: str,
        meta_columns: Optional[List[str]] = None,
        use_index_as_id: bool = False,
    ):
        """
        Create a PandasDataFrameConverter component.

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
        if self.use_index_as_id:
            if isinstance(dataframe.index, pd.MultiIndex):
                return False
        return True

    @component.output_types(documents=List[Document])
    def run(
        self,
        dataframe: pd.DataFrame,
        meta: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    ):
        """
        Converts text files to Documents.

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
        index_col_name = "index"

        meta_list = normalize_metadata(meta, sources_count=dataframe.shape[0])

        selected_columns = [self.content_column, *self.meta_columns]

        data_rows = dataframe[selected_columns].to_dict(orient="records")
        if self.use_index_as_id:
            indexes = dataframe.index.to_list()
            data_rows = [{index_col_name: str(idx), **row} for idx, row in zip(indexes, data_rows)]

        documents = []
        for i, row in enumerate(data_rows):
            doc_id = row.pop(index_col_name) if self.use_index_as_id else None
            content = row.pop(self.content_column)
            meta_row = {k: v for k, v in row.items() if k in self.meta_columns} if self.meta_columns else {}
            metadata = {**meta_row, **meta_list[i]} if meta_list else meta_row
            doc = Document(id=doc_id, content=content, meta=metadata)
            documents.append(doc)

        return {"documents": documents}
