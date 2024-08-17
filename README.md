# DiskVec

DiskVec is a Python package for efficient storage and retrieval of high-dimensional vectors on disk. It implements a tree-based index structure for fast approximate nearest neighbor search.

## Features

- Disk-based storage for large vector datasets
- Fast approximate nearest neighbor search
- Support for custom vector dimensions and data types
- Efficient batch insertion of vectors

## Installation

```bash
pip install git+https://github.com/jpmag7/DiskVec.git
```

## Quick Start

```python
import numpy as np
from diskvec import DiskVec

# Initialize DiskVec
index = DiskVec("my_index", dim=128, dtype=np.float64)

# Insert vectors
vectors = np.random.rand(1000, 128).astype(np.float64)
values = np.arange(1000)
index.insert(vectors, values)

# Search
query = np.random.rand(128).astype(np.float64)
value, vector, distance = index.search(query)

# Close the index
index.close()
```

## API Reference

### DiskVec

```python
DiskVec(file: str, dim: int, efc: int = 32, dtype: np.dtype = np.float64, ...)
```

- `file`: Path to the index file
- `dim`: Dimension of the vectors
- `efc`: Number of children per node in the tree
- `dtype`: Data type of the vectors
- `overwrite`: Whether to overwrite an existing index
- `chunk_size`: Number of vectors to process at once
- `max_map_step`: Maximum size increment in bytes of the data file
- `id_bytes`: Number of bytes to represent vector IDs
- `values_dtype`: Data type of the value elements

### Methods

- `insert(vecs: np.array, values: np.array)`: Insert vectors and their corresponding values
- `search(query: np.array)`: Search for the closest vector to the query
- `close()`: Close the index
- `delete()`: Delete the index from the file system

## License

[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
