from cal256_dp import Caltech256
import zmq

if __name__ == "__main__":
    ctx = zmq.Context()
    skt = ctx.socket(zmq.REQ)
    skt.connect("tcp://localhost:3333")

    dp = Caltech256("/home/jcsu/Dev/motivation/dataset/256_ObjectCategories")
    skt.send_pyobj(dp)
    skt.recv_pyobj()