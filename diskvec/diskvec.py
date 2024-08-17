import numpy as np
import lmdb
import threading
import shutil
import json
import os


class DiskVec:

    dtype_to_string : dict = {
        np.bool_: 'bool',
        np.int8: 'int8',
        np.int16: 'int16',
        np.int32: 'int32',
        np.int64: 'int64',
        np.uint8: 'uint8',
        np.uint16: 'uint16',
        np.uint32: 'uint32',
        np.uint64: 'uint64',
        np.float16: 'float16',
        np.float32: 'float32',
        np.float64: 'float64',
        np.complex64: 'complex64',
        np.complex128: 'complex128',
        np.str_: 'str',
        np.datetime64: 'datetime64',
        np.timedelta64: 'timedelta64',
        np.object_: 'object'
    }
    string_to_dtype : dict = {v: k for k, v in dtype_to_string.items()}
    
    def __init__(
        self,
        file:str,
        dim:int,
        efc:int=32,
        dtype:np.dtype=np.float64,
        overwrite:bool=False,
        chunk_size:int=100_000,
        max_map_step:int=1*1024**3, # 1GB
        id_bytes:int=5,
        values_dtype:np.dtype=np.int64,
    ) -> None:
        """
        :param str file: the path to the index
        :param int dim: dimension of the vectors
        :param int efc: number of childs per node in the tree
        :param np.dtype dtype: dtype of the vector (will cast)
        :param bool overwrite: overwrite existing index
        :param int chunk_size: number of vectors to process at the same time
        :param int max_map_step: max size increment of the data file 
        :param int id_bytes: number of bytes to represent the vector ids
        :param np.dtype values_dtype: dtype of the value elements
        :raises ValueError: if the chunk_size is less than efc
        """
        self.file = file
        self.dim = dim
        self.efc = efc
        self.dtype = dtype
        self.chunk_size = chunk_size
        self.vec_count = 0
        self.layer_count = 0
        self.max_map_step = max_map_step
        self.values_dtype = values_dtype
        self._id_size = id_bytes
        self._dtype_size = np.dtype(self.dtype).itemsize
        self._values_size = np.dtype(self.values_dtype).itemsize
        if self.chunk_size < self.efc:
            raise ValueError(f"chunk_size must be >= to efc. chunk_size={self.chunk_size} - efc={self.efc}")
        lock_path = os.path.join(self.file, "lock.mdb")
        if os.path.exists(lock_path):
            os.remove(lock_path) # Remove lock to avoid error on reopen
        if overwrite:
            self.delete()
        exists = os.path.exists(self.file)
        self._env = lmdb.open(self.file, map_size=10485760)  # 10 MB
        if not exists:
            self._set_childs(0, []) # Initialize if its new
            self._save_metadata()
        else:
            self._load_metadata()
    
    
    def _save_metadata(self):
        with open(os.path.join(self.file, "metadata.json"), "w") as f:
            data = {
                "dim" : self.dim,
                "efc" : self.efc,
                "dtype" : DiskVec.dtype_to_string[self.dtype],
                "chunk_size" : self.chunk_size,
                "vec_count" : self.vec_count,
                "layer_count" : self.layer_count,
                "max_map_step" : self.max_map_step,
                "values_dtype" : DiskVec.dtype_to_string[self.values_dtype],
                "_id_size" : self._id_size,
                "_dtype_size" : self._dtype_size,
                "_values_size" : self._values_size,
            }
            f.write(json.dumps(data))
        
    
    def _load_metadata(self):
        with open(os.path.join(self.file, "metadata.json"), "r") as f:
            data = json.loads(f.read())
        data["dtype"] = DiskVec.string_to_dtype[data["dtype"]]
        data["values_dtype"] = DiskVec.string_to_dtype[data["values_dtype"]]
        for k, v in data.items():
            setattr(self, k, v)


    def _get_childs(self, n_id:int):
        with self._env.begin(write=False) as txn:
            encoded_data = txn.get(n_id.to_bytes(self._id_size, 'little')) or []
        decoded_data = []
        offset = 0
        while offset < len(encoded_data):
            num = int.from_bytes(encoded_data[offset:offset + self._id_size], 'little')
            offset += self._id_size
            value = np.frombuffer(encoded_data[offset:offset + self._values_size], dtype=self.values_dtype)[0]
            offset += self._values_size
            array = np.frombuffer(encoded_data[offset:offset + self.dim * self._dtype_size], dtype=self.dtype)
            offset += self.dim * self._dtype_size
            decoded_data.append((num, value, array))
        return decoded_data


    def _set_childs(self, n_id:int, n_childs:list):
        encoded_data = b''
        for num, value, array in n_childs:
            encoded_data += num.to_bytes(self._id_size, 'little') # Encode id
            encoded_data += value.tobytes()                       # Encode value
            encoded_data += array.tobytes()                       # Encode array
        try:
            with self._env.begin(write=True) as txn:
                txn.put(n_id.to_bytes(self._id_size, 'little'), encoded_data)
        except lmdb.MapFullError:
            map_size = self._env.info()['map_size']
            map_size += min(map_size, self.max_map_step)
            self._env.set_mapsize(map_size)
            with self._env.begin(write=True) as txn:
                txn.put(n_id.to_bytes(self._id_size, 'little'), encoded_data)
            

    def _search(self, vec:np.array, ep_id:int=0, ep_value:np.integer=None, ep_vec:np.array=None):
        ep_childs = self._get_childs(ep_id)
        if len(ep_childs) == 0:
            return ep_value, ep_vec, np.linalg.norm(vec - ep_vec) if ep_vec is not None else float("+inf")
        min_dist = float("+inf")
        min_id = None
        min_vec = None
        min_value = None
        for c_id, c_value, c_vec in ep_childs:
            dist = np.linalg.norm(vec - c_vec)
            if dist < min_dist:
                min_dist = dist
                min_id = c_id
                min_vec = c_vec
                min_value = c_value
        min_value, min_vec, min_dist = self._search(vec, min_id, min_value, min_vec)
        d = np.linalg.norm(vec - ep_vec) if ep_vec is not None else float("+inf")
        if d <= min_dist:
            return ep_value, ep_vec, d
        else:
            return min_value, min_vec, min_dist
    
    
    def search(self, query:np.array):
        """
        Searchs for the closes value and vector to a given query

        :param np.array query: the vector query to be searched
        :return: closest value, closest vector, closest distance
        :raises ValueError: if query shape is different from index dimension (dim,)
        """
        if query.shape != (self.dim,):
            raise ValueError(f"Query shape is incorrect. Expected ({self.dim},), but got {query.shape}")
        if query.dtype != self.dtype:
            query = query.astype(self.dtype)
        return self._search(query)


    def _insert(self, vecs:np.array, values:np.array, ep_id:int=0, is_new:bool=False, layer:int=1):
        ep_childs = [] if is_new else self._get_childs(ep_id)
        num_childs = len(ep_childs)
        setter_thread = None

        if num_childs < self.efc:
            # Add remaining vectors as new nodes if there is space
            add_count = min(self.efc - num_childs, vecs.shape[0])
            for vec, value in zip(vecs[:add_count], values[:add_count]):
                self.vec_count += 1
                ep_childs.append((self.vec_count, value, vec))
            setter_thread = threading.Thread(target=self._set_childs, args=(ep_id, ep_childs))
            setter_thread.start()
            vecs = vecs[add_count:]
            values = values[add_count:]

        # Process remaining vectors in chunks
        used_nodes = set()
        for start in range(0, vecs.shape[0], self.chunk_size):
            vec_chunk = vecs[start:start + self.chunk_size]
            val_chunk = values[start:start + self.chunk_size]

            children_vecs = np.array([c_vec for c_id, c_value, c_vec in ep_childs])  # All child vectors in an array

            # Calculate all pairwise distances between vec_chunk and children_vecs
            dists = np.linalg.norm(vec_chunk[:, None] - children_vecs[None, :], axis=2)  # shape: (len(vec_chunk), len(ep.childs))

            # Find the closest child for each vector in vec_chunk
            min_indices = np.argmin(dists, axis=1)  # shape: (len(vec_chunk),)

            # Group vectors by their closest child node
            for i in range(len(ep_childs)):
                vec_batch_for_child = vec_chunk[min_indices == i]
                val_batch_for_child = val_chunk[min_indices == i]
                if len(vec_batch_for_child) > 0:
                    # Recursively insert each batch into the corresponding child
                    self._insert(
                        vec_batch_for_child,
                        val_batch_for_child,
                        ep_id=ep_childs[i][0],
                        is_new=i+1>num_childs and not i in used_nodes,
                        layer=layer+1
                    )
                    used_nodes.add(i)
        if setter_thread is not None:
            setter_thread.join()
        if layer > self.layer_count: self.layer_count = layer
    
    
    def insert(self, vecs:np.array, values:np.array):
        """
        Inserts an array of vectors and values into the index

        :param np.array vecs: the vectors to be inserted with shape (num_vecs, dim)
        :param np.array values: the values corresponding to each vector with shape (num_vecs,)
        :raises ValueError: if vecs hasn't 2 dimensions
        :raises ValueError: if each vector doesn't have size dim
        :raises ValueError: if values don't have 1 dimension
        :raises ValueError: if number of vecs is different from number of values
        """
        if vecs.ndim != 2:
            raise ValueError(f"Vecs must have 2 dimensions. Got shape {vecs.shape}")
        if vecs.shape[1] != self.dim:
            raise ValueError(f"Each vec must have size {self.dim}. Got size {vecs.shape[1]}")
        if values.ndim != 1:
            raise ValueError(f"Values must have 1 dimension. Got shape {values.shape}")
        if values.shape[0] != vecs.shape[0]:
            raise ValueError(f"Number of vecs and values must be the same. Got {vecs.shape[0]} vecs and {values.shape[0]} values")
        if values.dtype != self.values_dtype:
            #raise ValueError(f"Values must have dtype {self.values_dtype}. Got dtype {values.dtype}")
            values = values.astype(self.values_dtype)
        if vecs.dtype != self.dtype:
            vecs = vecs.astype(self.dtype)
        permutation = np.random.permutation(vecs.shape[0])
        vecs = vecs[permutation]
        values = values[permutation]
        self._insert(vecs, values)
        self._save_metadata()
    
    
    def close(self):
        """
        Closes access to the index file
        """
        try:
            self._env.close()
        except Exception:
            pass
    
    
    def delete(self):
        """
        Deletes the index from the file system
        """
        self.close()
        try:
            shutil.rmtree(self.file)
        except Exception:
            pass
