import os
import zmq

from torchdata.dataloader2 import MultiProcessingReadingService
from torchdata.remoteloader import RemoteDataloader, Worker


if __name__ == "__main__":
    data_path = "/home/jcsu/Dev/motivation/dataset/256_ObjectCategories"
    os.environ["CONTROLLER_IP"] = "localhost"
    os.environ["CONTROLLER_PORT"] = "5556"
    
    ctx = zmq.Context()
    skt = ctx.socket(zmq.REQ)
    
    skt.connect(f"tcp://{os.environ['CONTROLLER_IP']}:{os.environ['CONTROLLER_PORT']}")
    skt.send_pyobj({"type": "get_dp", "data_path": data_path})

    rs = MultiProcessingReadingService(num_workers=4)
    dl = RemoteDataloader(dp, reading_service=rs, node_type=Worker)

    for idx, i in enumerate(dl):
        print(f"sending {idx}, batch_size: {len(i)}")
        dl.send_to_trainer(i)