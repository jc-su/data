import hashlib
import torch
import redis
import dill
from filelock import FileLock

def write_file(filepath, obj):
    with FileLock(filepath + ".lock"):
        torch.save(obj, filepath)

# Create the Redis connection
def put_redis_cache():
    r = redis.Redis(host='localhost', port=6379, db=0)

    test_cached_key = [hashlib.md5(str(i).encode()).hexdigest() for i in range(0, 100000, 30)]

    # Generate random 3, 224, 224 tensors and store them in Redis
    for key in test_cached_key:
        tensor = torch.randn(3, 224, 224)  # Create a random tensor
        tensor_bytes = dill.dumps(tensor)  # Serialize the tensor to bytes
        r.set(key, tensor_bytes)  # Store the bytes in Redis


def put_nvme_cache():
    test_cached_key = [hashlib.md5(str(i).encode()).hexdigest() for i in range(100000, 200000, 30)]
    for key in test_cached_key:
        tensor = torch.randn(3, 224, 224)
        path = "/home/jcsu/Dev/nvme_cache/" + key
        # print(path)
        write_file(path, tensor)

def remove_redis_cache():
    r = redis.Redis(host='localhost', port=6379, db=0)
    test_cached_key = [hashlib.md5(str(i).encode()).hexdigest() for i in range(0, 100000, 30)]
    for key in test_cached_key:
        r.delete(key)

if __name__ == '__main__':
    # remove_redis_cache()
    put_redis_cache()