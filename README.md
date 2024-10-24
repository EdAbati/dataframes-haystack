# Dataframes Haystack

[![PyPI - Version](https://img.shields.io/pypi/v/dataframes-haystack)](https://pypi.org/project/dataframes-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/dataframes-haystack?logo=python&logoColor=white)](https://pypi.org/project/dataframes-haystack)
[![PyPI - License](https://img.shields.io/pypi/l/dataframes-haystack)](https://pypi.org/project/dataframes-haystack)


[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

[![GH Actions Tests](https://github.com/EdAbati/dataframes-haystack/actions/workflows/test.yml/badge.svg)](https://github.com/EdAbati/dataframes-haystack/actions/workflows/test.yml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/EdAbati/dataframes-haystack/main.svg)](https://results.pre-commit.ci/latest/github/EdAbati/dataframes-haystack/main)

-----

## 📃 Description

`dataframes-haystack` is an extension for [Haystack 2](https://docs.haystack.deepset.ai/docs/intro) that enables integration with dataframe libraries.

The dataframe libraries currently supported are:
- [pandas](https://pandas.pydata.org/)
- [Polars](https://pola.rs)

The library offers various custom [Converters](https://docs.haystack.deepset.ai/docs/converters) components to transform dataframes into Haystack [`Document`](https://docs.haystack.deepset.ai/docs/data-classes#document) objects:
- `DataFrameFileToDocument` is a main generic converter that reads files using a dataframe backend and converts them into `Document` objects.
- `FileToPandasDataFrame` and `FileToPolarsDataFrame` read files and convert them into dataframes.
- `PandasDataFrameConverter` or `PolarsDataFrameConverter` convert data stored in dataframes into Haystack `Document`objects.

`dataframes-haystack` supports reading files in various formats:
- _csv_, _json_, _parquet_, _excel_, _html_, _xml_, _orc_, _pickle_, _fixed-width format_ for `pandas`. See the [pandas documentation](https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html) for more details.
- _csv_, _json_, _parquet_, _excel_, _avro_, _delta_, _ipc_ for `polars`. See the [polars documentation](https://docs.pola.rs/api/python/stable/reference/io.html) for more details.

## 🛠️ Installation

```sh
# for pandas (pandas is already included in `haystack-ai`)
pip install dataframes-haystack

# for polars
pip install "dataframes-haystack[polars]"
```

## 💻 Usage

> [!TIP]
> See the [Example Notebooks](./notebooks) for complete examples.

## DataFrameFileToDocument

[Complete example](https://github.com/EdAbati/dataframes-haystack/blob/main/notebooks/dataframe-file-to-doc-example.ipynb)

You can leverage both `pandas` and `polars` backends (thanks to [`narwhals`](https://github.com/narwhals-dev/narwhals)) to read your data!

```python
from dataframes_haystack.components.converters import DataFrameFileToDocument

converter = DataFrameFileToDocument(content_column="text_str")
documents = converter.run(files=["file1.csv", "file2.csv"])
```

```python
>>> documents
{'documents': [
    Document(id=0, content: 'Hello world', meta: {}),
    Document(id=1, content: 'Hello everyone', meta: {})
]}
```

### Pandas

[Complete example](https://github.com/EdAbati/dataframes-haystack/blob/main/notebooks/pandas-example.ipynb)

#### FileToPandasDataFrame

```python
from dataframes_haystack.components.converters.pandas import FileToPandasDataFrame

converter = FileToPandasDataFrame(file_format="csv")

output_dataframe = converter.run(
    file_paths=["data/doc1.csv", "data/doc2.csv"]
)
```

Result:
```python
>>> output_dataframe
{'dataframe': <pandas.DataFrame>}
```

#### PandasDataFrameConverter

```python
import pandas as pd

from dataframes_haystack.components.converters.pandas import PandasDataFrameConverter

df = pd.DataFrame({
    "text": ["Hello world", "Hello everyone"],
    "filename": ["doc1.txt", "doc2.txt"],
})

converter = PandasDataFrameConverter(content_column="text", meta_columns=["filename"])
documents = converter.run(df)
```

Result:
```python
>>> documents
{'documents': [
    Document(id=0, content: 'Hello world', meta: {'filename': 'doc1.txt'}),
    Document(id=1, content: 'Hello everyone', meta: {'filename': 'doc2.txt'})
]}
```

### Polars

[Complete example](https://github.com/EdAbati/dataframes-haystack/blob/main/notebooks/polars-example.ipynb)

#### FileToPolarsDataFrame

```python
from dataframes_haystack.components.converters.polars import FileToPolarsDataFrame

converter = FileToPolarsDataFrame(file_format="csv")

output_dataframe = converter.run(
    file_paths=["data/doc1.csv", "data/doc2.csv"]
)
```

Result:
```python
>>> output_dataframe
{'dataframe': <polars.DataFrame>}
```

#### PolarsDataFrameConverter

```python
import polars as pl

from dataframes_haystack.components.converters.polars import PolarsDataFrameConverter

df = pl.DataFrame({
    "text": ["Hello world", "Hello everyone"],
    "filename": ["doc1.txt", "doc2.txt"],
})

converter = PolarsDataFrameConverter(content_column="text", meta_columns=["filename"])
documents = converter.run(df)
```

Result:
```python
>>> documents
{'documents': [
    Document(id=0, content: 'Hello world', meta: {'filename': 'doc1.txt'}),
    Document(id=1, content: 'Hello everyone', meta: {'filename': 'doc2.txt'})
]}
```

## 🤝 Contributing

Do you have an idea for a new feature? Did you find a bug that needs fixing?

Feel free to [open an issue](https://github.com/EdAbati/dataframes-haystack/issues) or submit a PR!

### Setup development environment

Requirements: [`hatch`](https://hatch.pypa.io/latest/install/), [`pre-commit`](https://pre-commit.com/#install)

1. Clone the repository
1. Run `hatch shell` to create and activate a virtual environment
1. Run `pre-commit install` to install the pre-commit hooks. This will force the linting and formatting checks.

### Run tests

- Linting and formatting checks: `hatch run lint:fmt`
- Unit tests: `hatch run test-cov-all`

## ✍️ License

`dataframes-haystack` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
