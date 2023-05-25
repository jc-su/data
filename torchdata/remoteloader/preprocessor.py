import base64
import os

import zmq

from torchdata.dataloader2.graph._serialization import deserialize_datapipe
from torchdata.remoteloader import RemoteDataloader

if __name__ == "__main__":
    controller_ip = os.environ.get("CONTROLLER_IP")
    controller_port = os.environ.get("CONTROLLER_PORT")
    train_ip = os.environ.get("TRAIN_IP")
    train_port = os.environ.get("TRAIN_PORT")
    dp = deserialize_datapipe(base64.b64decode(os.environ.get("DP").encode("utf-8")))
    dl = RemoteDataloader(dp, node_type="PREPROCESSOR")

    for i in dl:
        pass
