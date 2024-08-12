import numpy as np
import os
import mmap
import struct
import heapq
from concurrent.futures import ThreadPoolExecutor
import pickle

class DiskHNSW:
    def __init__(self, index_folder, dim, max_elements, ef_construction=200, M=16, num_layers=4):
        self.index_folder = index_folder
        self.dim = dim
        self.max_elements = max_elements
        self.ef_construction = ef_construction
        self.M = M
        self.num_layers = num_layers
        self.cur_element_count = 0
        self.entry_point = None
        
        self.vector_filename = "vectors.dat"
        self.graph_filename = "graph.dat"
        self.metadata_filename = "metadata.pkl"
        
        self.layers = [[] for _ in range(num_layers)]
        self.executor = ThreadPoolExecutor(max_workers=os.cpu_count())

        if os.path.exists(os.path.join(index_folder, self.metadata_filename)):
            self._load()
        else:
            self._initialize_storage()

    def _initialize_storage(self):
        os.makedirs(self.index_folder, exist_ok=True)
        
        vector_path = os.path.join(self.index_folder, self.vector_filename)
        self.vector_file = np.memmap(vector_path, dtype='float32', mode='w+', shape=(self.max_elements, self.dim))
        
        graph_path = os.path.join(self.index_folder, self.graph_filename)
        graph_file_size = self.max_elements * self.M * 4 * self.num_layers
        with open(graph_path, "wb") as f:
            f.write(b'\0' * graph_file_size)
        self.graph_file = open(graph_path, "r+b")
        self.graph_mmap = mmap.mmap(self.graph_file.fileno(), 0)
        
        self._save_metadata()

    def _save_metadata(self):
        metadata = {
            'dim': self.dim,
            'max_elements': self.max_elements,
            'ef_construction': self.ef_construction,
            'M': self.M,
            'num_layers': self.num_layers,
            'cur_element_count': self.cur_element_count,
            'entry_point': self.entry_point,
            'layers': self.layers
        }
        metadata_path = os.path.join(self.index_folder, self.metadata_filename)
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)

    def _load(self):
        metadata_path = os.path.join(self.index_folder, self.metadata_filename)
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        self.dim = metadata['dim']
        self.max_elements = metadata['max_elements']
        self.ef_construction = metadata['ef_construction']
        self.M = metadata['M']
        self.num_layers = metadata['num_layers']
        self.cur_element_count = metadata['cur_element_count']
        self.entry_point = metadata['entry_point']
        self.layers = metadata['layers']
        
        vector_path = os.path.join(self.index_folder, self.vector_filename)
        self.vector_file = np.memmap(vector_path, dtype='float32', mode='r+', shape=(self.max_elements, self.dim))
        
        graph_path = os.path.join(self.index_folder, self.graph_filename)
        self.graph_file = open(graph_path, "r+b")
        self.graph_mmap = mmap.mmap(self.graph_file.fileno(), 0)

    def add_item(self, vector):
        if self.cur_element_count >= self.max_elements:
            raise Exception("Cannot add more elements, capacity reached")
        
        self.vector_file[self.cur_element_count] = vector
        
        layer = self._get_random_level()
        
        if self.cur_element_count == 0:
            self.entry_point = self.cur_element_count
            for l in range(layer + 1):
                self.layers[l].append(self.cur_element_count)
            self.cur_element_count += 1
            self._save_metadata()
            return
        
        ep = self.entry_point
        for l in range(layer, -1, -1):
            W = self._search_layer(vector, self.ef_construction, ep, l)
            neighbors = self._select_neighbors(self.cur_element_count, W, self.M, l, vector)
            self._add_connections(self.cur_element_count, neighbors, l)
            if W:
                ep = W[0][1]

        for l in range(layer + 1):
            self.layers[l].append(self.cur_element_count)
        
        if layer > self._get_element_level(self.entry_point):
            self.entry_point = self.cur_element_count
        
        self.cur_element_count += 1
        self._save_metadata()

    def _get_random_level(self):
        level = 0
        while np.random.random() < 0.5 and level < self.num_layers - 1:
            level += 1
        return level

    def _get_element_level(self, element):
        for i in range(self.num_layers - 1, -1, -1):
            if element in self.layers[i]:
                return i
        return 0

    def _search_layer(self, q, ef, ep, layer):
        v = set([ep])
        candidates = [(self._distance(q, self.vector_file[ep]), ep)]
        heapq.heapify(candidates)
        
        W = [(self._distance(q, self.vector_file[ep]), ep)]
        
        while candidates:
            c_dist, c = heapq.heappop(candidates)
            f_dist, _ = W[0]
            
            if c_dist > f_dist:
                break
            
            for e in self._get_neighbors(c, layer):
                if e == -1:  # No more neighbors
                    break
                if e not in v:
                    v.add(e)
                    e_dist = self._distance(q, self.vector_file[e])
                    
                    if e_dist < f_dist or len(W) < ef:
                        heapq.heappush(candidates, (e_dist, e))
                        heapq.heappush(W, (e_dist, e))
                        
                        if len(W) > ef:
                            heapq.heappop(W)
        
        return W

    def _select_neighbors(self, q, candidates, M, layer, vector):
        # Simple heuristic: select M closest neighbors
        return sorted(candidates, key=lambda x: x[0])[:M]

    def _add_connections(self, q, connections, layer):
        # Write connections to disk
        offset = q * self.M * 4 * self.num_layers + layer * self.M * 4
        for i, (_, c) in enumerate(connections):
            self.graph_mmap[offset + i * 4 : offset + (i + 1) * 4] = struct.pack('i', c)
        
        # Add backward connections
        for _, c in connections:
            self._add_backward_connection(c, q, layer)

    def _add_backward_connection(self, source, target, layer):
        offset = source * self.M * 4 * self.num_layers + layer * self.M * 4
        for i in range(self.M):
            neighbor = struct.unpack('i', self.graph_mmap[offset + i * 4 : offset + (i + 1) * 4])[0]
            if neighbor == -1:
                self.graph_mmap[offset + i * 4 : offset + (i + 1) * 4] = struct.pack('i', target)
                break
            elif neighbor == target:
                break

    def _get_neighbors(self, q, layer):
        offset = q * self.M * 4 * self.num_layers + layer * self.M * 4
        neighbors = []
        for i in range(self.M):
            neighbor = struct.unpack('i', self.graph_mmap[offset + i * 4 : offset + (i + 1) * 4])[0]
            if neighbor == -1:
                break
            neighbors.append(neighbor)
        return neighbors

    def search(self, vector, k=1):
        ep = self.entry_point
        for l in range(self._get_element_level(ep), -1, -1):
            W = self._search_layer(vector, self.ef_construction, ep, l)
            ep = W[0][1]
        
        # Sort by distance and return indices and distances
        results = sorted(W, key=lambda x: x[0])[:k]
        return [(int(idx), float(dist)) for dist, idx in results]

    def _distance(self, a, b):
        return np.linalg.norm(a - b)

    def __del__(self):
        if hasattr(self, 'vector_file'):
            del self.vector_file
        if hasattr(self, 'graph_mmap'):
            self.graph_mmap.close()
        if hasattr(self, 'graph_file'):
            self.graph_file.close()
        if hasattr(self, 'executor'):
            self.executor.shutdown()

    @classmethod
    def load(cls, folder_path):
        if not os.path.exists(os.path.join(folder_path, "metadata.pkl")):
            raise FileNotFoundError(f"No index found at {folder_path}")
        return cls(folder_path, dim=0, max_elements=0)  # Dummy values, will be overwritten in _load
