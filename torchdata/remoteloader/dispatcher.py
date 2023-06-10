import os
import zmq
import redis
from itertools import zip_longest
import dill
import uuid
from itertools import cycle
from collections import Counter
from filelock import FileLock


from collections import OrderedDict


class LRU_Cache:
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key):
        if key in self.cache:
            # Move the accessed item to the end of the cache
            self.cache.move_to_end(key)
            return self.cache[key]
        else:
            return None

    def put(self, key, value):
        if key in self.cache:
            # Move the updated item to the end of the cache
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            # Remove the least recently used item
            self.cache.popitem(last=False)


class Dispatcher:
    def __init__(self, massage):
        self.massage = massage
        self.total_batch_size = sum(worker['batch_size'] for worker in test_message['preprocessing_services'].values())
        ctx = zmq.Context()
        self.skt = ctx.socket(zmq.REQ)
        self.skt.connect(f"tcp://localhost:2000")
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)

    def dispatch_to_trainer(self, trainer, files):
        pass

    def dispatch_to_worker(self):
        pass

    def redis_cache_exists(self, keys):
        pipe = self.redis_client.pipeline()
        existing_keys = []
        found_keys = set()
        key_counts = Counter(keys)
        key_batch = []

        for key in keys:
            if key not in found_keys:
                pipe.exists(key)
                key_batch.append(key)
            if len(pipe.command_stack) >= self.total_batch_size:
                results = pipe.execute()
                for k, v in zip(key_batch, results):
                    if v:
                        found_keys.add(k)
                        existing_keys.extend([k] * key_counts[k])
                while len(existing_keys) >= self.total_batch_size:
                    yield existing_keys[:self.total_batch_size]
                    existing_keys = existing_keys[self.total_batch_size:]
                pipe = self.redis_client.pipeline()
                key_batch = []

        # Execute remaining commands in the pipeline
        if pipe.command_stack:
            results = pipe.execute()
            for k, v in zip(key_batch, results):
                if v:
                    found_keys.add(k)
                    existing_keys.extend([k] * key_counts[k])

        # Append remaining keys that have been found in Redis before
        for key in key_batch:
            if key in found_keys:
                existing_keys.extend([k] * key_counts[k])

        # Yield remaining keys in the existing_keys list
        while len(existing_keys) >= self.total_batch_size:
            yield existing_keys[:self.total_batch_size]
            existing_keys = existing_keys[self.total_batch_size:]

        # If there are less than batch_size keys left, yield them anyway
        if existing_keys:
            yield existing_keys

    def nvme_cache_exists(self, keys, directory):
        existing_objects = []
        object_cache = LRU_Cache(1000)  # Cache 10000 objects in memory
        key_counts = Counter(keys)

        for key in keys:
            obj = object_cache.get(key)
            if obj is not None:
                existing_objects.extend([obj] * key_counts[key])
            else:
                filepath = os.path.join(directory, key)
                if os.path.isfile(filepath):
                    print("reading from nvme")
                    # with open(filepath, 'rb') as file:
                    #     obj = dill.load(file)
                    #     object_cache.put(key, obj)
                    #     existing_objects.extend([obj] * key_counts[key])
                    obj = self.read_file(filepath)
                    object_cache.put(key, obj)
                    existing_objects.extend([obj] * key_counts[key])

            while len(existing_objects) >= self.total_batch_size:
                # print("yielding")
                yield existing_objects[:self.total_batch_size]
                existing_objects = existing_objects[self.total_batch_size:]

        if existing_objects:
            yield existing_objects

    def _sharding(self, uncache_files, workers):
        # Create a round-robin cycle of workers, where each worker is repeated according to its batch size
        worker_cycle = cycle(worker for worker in workers
                             for _ in range(workers[worker]['batch_size']))

        batches = {worker: [] for worker in workers}

        # Create a counter for each worker's batch
        batch_counts = {worker: 0 for worker in workers}

        for file_id in uncache_files:
            worker = next(worker_cycle)
            if not batches[worker] or batch_counts[worker] == workers[worker]['batch_size']:
                batches[worker].append([])
                batch_counts[worker] = 0
            batches[worker][-1].append(file_id)
            batch_counts[worker] += 1

        return batches

    def read_file(self, filepath):
        with FileLock(filepath + ".lock"):
            with open(filepath, 'rb') as file:
                obj = dill.load(file)
        return obj


# Example usage:
if __name__ == '__main__':
    import hashlib
    id_set = [hashlib.md5(str(i).encode()).hexdigest() for i in range(100000, 150000, 30)]
    file_hash_list = id_set * 2
    test_message = {
        "file_hash_list": file_hash_list,
        "preprocessing_services": {
            "worker_1": {
                "batch_size": 30,
                "number_of_process": 1
            },
            "worker_2": {
                "batch_size": 20,
                "number_of_process": 1
            },
            "worker_3": {
                "batch_size": 50,
                "number_of_process": 2
            }
        },
    }

    # random choose 3333 files to be cached
    test_cached = [hashlib.md5(str(i).encode()).hexdigest() for i in range(0, 100000, 30)]

    print(f"Total number of files: {len(test_message['file_hash_list'])}, searching for {len(test_cached)} files")

    dispat = Dispatcher(test_message)
    # b = dispat._sharding(test_cached, test_message['preprocessing_services'])
    # for i in b.keys():
    #     print(i, len(b[i][-1]))

    import time
    start_time = time.time()
    for i in dispat.nvme_cache_exists(file_hash_list, '/home/jcsu/Dev/nvme_cache'):
        print(len(i), len(set(i)))
    print(f"Time elapsed: {time.time() - start_time}")
