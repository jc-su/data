import os
import zmq

from torchdata.dataloader2 import (DataLoader2, MultiProcessingReadingService)
from torchdata.dataloader2.adapter import Shuffle, Batch
from torchdata.remoteloader import RemoteDataloader, Worker


if __name__ == "__main__":

    os.environ["CONTROLLER_IP"] = "localhost"
    os.environ["CONTROLLER_PORT"] = "5556"

    rs = MultiProcessingReadingService(num_workers=5)
    dl = RemoteDataloader(dp, [Shuffle(True)], node_type=Worker, reading_service=rs)


    # dl.seed = random.randint(0, 100)
    for idx, i in enumerate(dl):
        print(f"sending {idx}, batch_size: {len(i['id'])}")
        dl.send_to_trainer(i)