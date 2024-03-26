from typing import Any, Dict, List, Optional, Union

from haystack import Document, component, logging
from haystack.components.converters.utils import normalize_metadata
from haystack.lazy_imports import LazyImport

logger = logging.getLogger(__name__)

with LazyImport(message="Run 'pip install \"dataframes-haystack[polars]\"'") as polars_import:
    import polars as pl


@component
class PolarsDataFrameConverter:
    """
    Converts data in a polars.DataFrame to Documents.

    Usage example:
    ```python
    from dataframes_haystack.components.converters.polars import PolarsDataFrameConverter

    converter = PolarsDataFrameToDocument(content_column="text")
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
