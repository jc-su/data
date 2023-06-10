from ast import List
import os
from torch import Generator
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
    r"""
    LRU_Cache is a LRU cache that can be used in the dispatcher nvme cache.

    Args:
        capacity: The capacity of the cache.
    """

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
    r"""
    Dispatcher is a dispatcher that can be used to dispatch cache to trainer and shard data to workers.

    Args:
        message: The message to dispatch.
    """

    def __init__(self, massage):
        self.content = massage
        self.total_batch_size = sum(worker['batch_size'] for worker in test_message['preprocessing_services'].values())
        # ctx = zmq.Context()
        # self.skt = ctx.socket(zmq.PUSH)
        # self.skt.connect(f"tcp://localhost:2000")
        self.redis_client = redis.Redis(host="localhost", port=6379, db=0)
        self.nvme_cache_directory = os.environ.get("NVME_CACHE_DIRECTORY")

        self.redis_cache_uuid = f"redis-{uuid.uuid4()}".encode("utf-8")
        self.nvme_cache_uuid = f"nvme-{uuid.uuid4()}".encode("utf-8")

    def dispatch_to_trainer(self, drop_last=False):
        print(f"init file list lenght: {len(self.content['file_hash_list'])}")
        hitted_files_id, redis_remain_files = self.dispatch_cache_to_trainer(self.redis_cache_exists(self.content["file_hash_list"]), "redis")
        self.update_file_hash_list(hitted_files_id)

        print(f"remain file list lenght: {len(self.content['file_hash_list'])}")
        hitted_files_id, nvme_remain_files = self.dispatch_cache_to_trainer(self.nvme_cache_exists(self.content["file_hash_list"]), "nvme")
        self.update_file_hash_list(hitted_files_id)

        if not drop_last and len(redis_remain_files) + len(nvme_remain_files) >= self.total_batch_size:
            redis_remain_objects = self.redis_client.mget(redis_remain_files)
            last_batch = redis_remain_objects + nvme_remain_files
            last_batch = last_batch[:self.total_batch_size]
            print(f"Sending last batch to trainer, batch size: {len(last_batch)}")

    def dispatch_cache_to_trainer(self, cache_gen: Generator, cache_type: str) -> List:
        hitted_files_id = []
        remain_files = []
        for batch in cache_gen:
            hitted_files_id.extend(batch)
            if len(batch) < self.total_batch_size:
                remain_files.extend(batch)
                break
            print(f"Sending {cache_type} cache to trainer, batch size: {len(batch)}")
        return hitted_files_id, remain_files

    def update_file_hash_list(self, hitted_files_id: List) -> None:
        hitted_files_set = set(hitted_files_id)
        self.content["file_hash_list"] = [file for file in self.content["file_hash_list"] if file not in hitted_files_set]

    def dispatch_to_worker(self):
        shard = self._sharding(self.content["file_hash_list"])
        for worker_id, file_hash_list in shard.items():
            # self.skt.send_multipart([worker_id, dill.dumps(file_hash_list)])
            print(f"Sending batch to worker{worker_id}, batch size: {len(file_hash_list[0])}, iteration number: {len(file_hash_list)}")

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

    def nvme_cache_exists(self, keys):
        existing_objects = []
        object_cache = LRU_Cache(1000)  # Cache 10000 objects in memory
        key_counts = Counter(keys)

        for key in keys:
            obj = object_cache.get(key)
            if obj is not None:
                existing_objects.extend([obj] * key_counts[key])
            else:
                filepath = os.path.join(self.nvme_cache_directory, key)
                if os.path.isfile(filepath):
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

    def _sharding(self, uncache_files):
        workers = self.content["preprocessing_services"]
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

    def start_dispatching(self):
        self.dispatch_to_trainer()
        self.dispatch_to_worker()


# Example usage:
if __name__ == '__main__':
    os.environ["NVME_CACHE_DIRECTORY"] = "/home/jcsu/Dev/nvme_cache"
    import hashlib
    id_set = [hashlib.md5(str(i).encode()).hexdigest() for i in range(0, 200000, 10)]
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
    redis_cached_key = [hashlib.md5(str(i).encode()).hexdigest() for i in range(0, 100000, 30)]
    nvme_cached_key = [hashlib.md5(str(i).encode()).hexdigest() for i in range(100000, 200000, 30)]

    print(len([i for i in file_hash_list if i in nvme_cached_key]))
    print(f"Total number of files: {len(test_message['file_hash_list'])}, searching for cached files in Redis with {len(redis_cached_key)} keys and in NVME with {len(nvme_cached_key)} keys")
    import time
    start = time.time()
    dispat = Dispatcher(test_message)
    dispat.start_dispatching()
    print(f"Total time: {time.time() - start}")
