from diskvec import DiskVec
import numpy as np
import time

dim = 128
size = 10_000
tests = 1_000
index = DiskVec("index", dim, overwrite=True)

vects = np.random.rand(size, dim)
vals  = np.arange(size)
start = time.time()
index.insert(vects, vals)

index = DiskVec("index", dim)
print("Index reloaded from disk")

print(f"Insertion time: {time.time() - start}s")
print(f"Layer count: {index.layer_count}")
print(f"Vector count: {index.vec_count}")


res1 = []
res2 = []
dists = []
times = []
for _ in range(tests):
    r = np.random.randint(0, vects.shape[0])
    vec = vects[r]
    val = vals[r]
    start = time.time()
    v, answer, dist = index.search(vec)
    times.append(time.time() - start)
    distances = np.linalg.norm(vects - vec, axis=1)
    min_index = np.argmin(distances)
    r1 = all(vects[min_index] == answer)
    r2 = v == val
    res1.append(1 if r1 else 0)
    res2.append(1 if r2 else 0)
    if r1: dists.append(dist)

if len(res1) > 0:
    print("Tests:", sum(res1))
    print("Vector accuracy:", sum(res1) / len(res1))
    print("Value accuracy:", sum(res2) / len(res2))
    print("Avg dist:", sum(dists) / max(len(dists), 1))
    print("Avg time:", sum(times) / max(len(times), 1))
