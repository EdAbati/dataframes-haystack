# Dataframes Haystack

[![PyPI - Version](https://img.shields.io/pypi/v/dataframes-haystack.svg)](https://pypi.org/project/dataframes-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/dataframes-haystack.svg)](https://pypi.org/project/dataframes-haystack)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/EdAbati/dataframes-haystack/main.svg)](https://results.pre-commit.ci/latest/github/EdAbati/dataframes-haystack/main)

-----

**Table of Contents**

- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Description

`dataframes-haystack` is a Python library that allows various dataframe libraries to integrate with [Haystack 2.x](https://docs.haystack.deepset.ai/docs/intro).

The library offers custom [Converters](https://docs.haystack.deepset.ai/docs/converters) components that convert data in dataframes into Haystack [`Document`s](https://docs.haystack.deepset.ai/docs/data-classes#document).

The dataframe libraries currently supported are:
- [Pandas](https://pandas.pydata.org/)
- [Polars](https://pola.rs)

## Installation

```sh
# for pandas (pandas is already included in `haystack-ai`)
pip install dataframes-haystack

# for polars
pip install "dataframes-haystack[polars]"
```

## Usage

### Pandas

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
{'documents': [Document(id=2eaefcdeb8d31f9f3d543c614233476ff70c0ed5aae609667172786d09588223, content: 'Hello world', meta: {'filename': 'doc1.txt'}), Document(id=bdc99cbfe819356159950dbaffa0521b47ec3ac2ff040604c93fe7798cc71efc, content: 'Hello everyone', meta: {'filename': 'doc2.txt'})]}
```

### Polars

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
{'documents': [Document(id=2eaefcdeb8d31f9f3d543c614233476ff70c0ed5aae609667172786d09588223, content: 'Hello world', meta: {'filename': 'doc1.txt'}), Document(id=bdc99cbfe819356159950dbaffa0521b47ec3ac2ff040604c93fe7798cc71efc, content: 'Hello everyone', meta: {'filename': 'doc2.txt'})]}
```

## Contributing

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

## License

`dataframes-haystack` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
