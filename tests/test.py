import numpy as np
from diskhnsw import DiskHNSW
import traceback

# Test parameters
dim = 128
max_elements = 10000
num_test_vectors = 1000
k = 5  # Number of nearest neighbors to retrieve

try:
    print("Creating index...")
    hnsw = DiskHNSW("hnsw_index", dim=dim, max_elements=max_elements)

    # Generate and add test vectors
    print(f"Adding {num_test_vectors} vectors to the index...")
    test_vectors = np.random.random((num_test_vectors, dim)).astype(np.float32)
    for i, vector in enumerate(test_vectors):
        hnsw.add_item(vector)
        if (i + 1) % 100 == 0:
            print(f"Added {i + 1} vectors")

    # Perform a search
    print("\nPerforming a search...")
    query_vector = np.random.random(dim).astype(np.float32)
    results = hnsw.search(query_vector, k=k)
    print("Search results before saving:")
    for idx, distance in results:
        print(f"Index: {idx}, Distance: {distance}")

    # Load the index
    print("\nLoading the index...")
    loaded_hnsw = DiskHNSW.load("hnsw_index")

    # Perform a search on the loaded index
    print("\nPerforming a search on the loaded index...")
    results = loaded_hnsw.search(query_vector, k=k)
    print("Search results after loading:")
    for idx, distance in results:
        print(f"Index: {idx}, Distance: {distance}")

    # Verify that the results are the same
    print("\nVerifying results...")
    original_indices = set(idx for idx, _ in results)
    loaded_indices = set(idx for idx, _ in loaded_hnsw.search(query_vector, k=k))
    if original_indices == loaded_indices:
        print("Test passed: Search results are consistent before and after saving/loading.")
    else:
        print("Test failed: Search results are different before and after saving/loading.")

    print("\nTest completed successfully.")

except Exception as e:
    print(f"An error occurred: {str(e)}")
    print(traceback.format_exc())
