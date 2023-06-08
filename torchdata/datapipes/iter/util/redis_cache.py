from typing import Optional, TypeVar, Iterator, TypeVar

from torch.utils.data.datapipes.utils.common import DILL_AVAILABLE

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe
import redis
if DILL_AVAILABLE:
    import dill

    dill.extend(use_dill=False)

T_co = TypeVar("T_co")


@functional_datapipe("redis_cache")
class RedisCacheHolderIterDataPipe(IterDataPipe[T_co]):
    def __init__(self, source_dp: IterDataPipe[T_co], redis_url: str, cached_elements: Optional[int] = None) -> None:
        self.source_dp: IterDataPipe[T_co] = source_dp
        self._client = redis.Redis.from_url(redis_url)
        self._key = "tpipe"
        self._start_idx = 0
        # use number of cached elements
        self.cached_elements = cached_elements

    def _iter_stored(self):
        for idx in range(0, self._cache_list_len()):
            yield self._deserialize(self._client.lindex(self._key, idx))

    def _deserialize(self, response):
        return dill.loads(response)

    def _serialize(self, value):
        return dill.dumps(value)

    def __iter__(self) -> Iterator[T_co]:
        if self._cache_list_len() > 1:
            for idx, data in enumerate(self.source_dp):
                print(data)
                if idx < self._start_idx:
                    yield data
                else:
                    break
            yield from self._iter_stored()
        else:
            for data in self.source_dp:
                self._client.rpush(self._key, self._serialize(data))

                # Cache reaches element limit
                if self.cached_elements is not None and self._cache_list_len() > self.cached_elements:
                    self._client.lpop(self._key)
                    self._start_idx += 1
                yield data

    def __contains__(self, key):
        return self._client.exists(key)

    def _cache_list_len(self):
        return self._client.llen(self._key)

    def __len__(self) -> int:
        try:
            return len(self.source_dp)
        except TypeError:
            # if list has been created in the database
            if self._key in self:
                return self._start_idx + self._cache_list_len()
            else:
                raise TypeError(f"{type(self).__name__} instance doesn't have valid length until the cache is loaded.")
