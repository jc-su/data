import base64
import os

import dill
import zmq

from torchdata.dataloader2 import (DataLoader2, MultiProcessingReadingService)
from torchdata.remoteloader import RemoteDataloader

if __name__ == "__main__":

    os.environ["PORT"] = "5555"
    os.environ["CONTROLLER_IP"] = "localhost"
    os.environ["CONTROLLER_PORT"] = "5556"
    os.environ["ID"] = "1"

    # trainer_ip = os.environ.get("TRAIN_IP")
    # trainer_port = os.environ.get("TRAIN_PORT")
    trainer_ip = "localhost"
    trainer_port = "5557"
    id = os.environ.get("ID")

    ctx = zmq.Context()
    preprocessor_skt = ctx.socket(zmq.PUSH)
    preprocessor_skt.connect(f"tcp://{trainer_ip}:{trainer_port}")

    controller_skt = ctx.socket(zmq.REQ)
    controller_skt.connect(f"tcp://{os.environ.get('CONTROLLER_IP')}:{os.environ.get('CONTROLLER_PORT')}")
    controller_skt.send_pyobj({"type": "new", "id": id})
    # dp = controller_skt.recv_pyobj()
    # res = controller_skt.recv_pyobj()
    # if res["status"] != "OK":
    #     raise RuntimeError("Controller error")
    # dp = dill.loads(res["datapipe"])
    import random
    import time

    import numpy as np
    import torch

    from torchdata.datapipes.iter import IterableWrapper
    dp = IterableWrapper(range(100 * 100)).shuffle().sharding_filter().map(lambda x: np.array([x, x + 1])).batch(20)
    # dl = RemoteDataloader(dp, node_type="PREPROCESSOR")
    dl = DataLoader2(dp, reading_service=MultiProcessingReadingService(num_workers=2))
    t = random.uniform(0.1, 0.5)
    dl.seed = random.randint(0, 100)
    for i in dl:
        print(t, i)
        preprocessor_skt.send_pyobj({"id": t, "data": i})
        time.sleep(t)
