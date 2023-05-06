import hashlib
import inspect
import os.path
import sys
import time
import uuid
import warnings

from collections import deque
from functools import partial
from typing import Any, Callable, Deque, Dict, Iterator, List, Optional, Tuple, TypeVar

try:
    import portalocker
except ImportError:
    portalocker = None

from torch.utils.data.datapipes.utils.common import _check_unpickable_fn, DILL_AVAILABLE

from torch.utils.data.graph import traverse_dps
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterableWrapper, IterDataPipe

if DILL_AVAILABLE:
    import dill

    dill.extend(use_dill=False)

@functional_datapipe("in_nvme_cache")
class InNVMeCacheHolderIterDataPipe(IterDataPipe[T_co]):
    pass
