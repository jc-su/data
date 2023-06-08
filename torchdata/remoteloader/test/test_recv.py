import os
import zmq
from torchdata.remoteloader import RemoteDataloader


if __name__ == "__main__":
    os.environ["CONTROLLER_IP"] = "localhost"
    os.environ["CONTROLLER_PORT"] = "5556"

    ctx = zmq.Context()
    skt = ctx.socket(zmq.REP)
    skt.bind(f"tcp://*:3333")

    skt2 = ctx.socket(zmq.REQ)
    skt2.connect(f"tcp://localhost:6666")

    dp = skt.recv_pyobj()
    skt.send_pyobj("ok")
    print("received dp")
    skt2.send_pyobj(dp)
        
    rdl = RemoteDataloader(dp)
    for epoch in range(2):
        print("epoch", epoch)
        for idx, i in enumerate(rdl):
            print(f"batch_size: {rdl.batch_size}")
            print("-----")