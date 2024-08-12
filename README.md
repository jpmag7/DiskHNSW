# DiskHNSW: Disk-Based Hierarchical Navigable Small World Index

DiskHNSW is a Python implementation of a disk-based Hierarchical Navigable Small World (HNSW) index for efficient approximate nearest neighbor search. This implementation is designed to handle large datasets that exceed available RAM by storing the index structure on disk.

## Key Features:
- Disk-based storage for scalability
- Efficient approximate nearest neighbor search
- Automatic saving and loading of index state
- Support for high-dimensional vector data
- Configurable index parameters (M, ef_construction, num_layers)

## Use Cases:
- Large-scale similarity search
- Recommendation systems
- Content-based retrieval
- Machine learning applications requiring fast nearest neighbor lookups

This repository includes the core DiskHNSW class implementation and a test script to demonstrate its usage and verify functionality.
