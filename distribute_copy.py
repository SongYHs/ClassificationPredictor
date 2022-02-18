# TODO 实现数据分发和数据写入

from multiprocessing import Process, Manager, Event
from utils import pack, unpack
import argparse
import queue
import os
import zmq
import trio
import time
from datetime import timedelta


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def close_write(files, files_dict, error_dict):
    done, none, error = [], [], []
    for f in files:
        if files_dict.get(f, -1) == 0:
            done.append(f)
        elif files_dict.get(f, -0.1) - error_dict.get(f, -0.01) == 0:
            error.append(f)
        else:
            none.append((f, files_dict.get(f, None)))

    return done, none, error


def write_fun(data_dir, out_name, result):
    with open(os.path.join(data_dir, out_name), 'ab+') as f:
        f.write(result)


async def read_async(data_iter, self):
    print(f"读{os.getpid()}")
    robot = self.read_socket.recv(copy=False)
    robot, routing_id = str(robot), robot.routing_id
    if robot == "request":
        try:
            data_id, data, filename, file_count = data_iter.__next__()
            # print(f"{routing_id}号机器请求数据{data_id}")
            task_id = filename + "#@#" + str(data_id)
            send_data = pack("data", task_id, data)
            self.read_socket.send(send_data, routing_id=routing_id)
            if file_count > -1:
                self.Q.put([filename, file_count])
        except StopIteration:
            send_data = pack("done", '', '')
            self.read_socket.send(send_data, routing_id=routing_id)

    elif robot == "exit":
        print("预测异常，退出分发服务")
        self.exit_flag.set()


async def write_async(self, files_dict, error_dict):
    print(f"写{os.getpid()}")
    if not self.Q.qsize() == 0:
        _data = self.Q.get_nowait()
        name, count = _data
        files_dict[name] = files_dict.get(name, 0) + count
    done, none, error = close_write(self.files, files_dict, error_dict)
    print(done, none, error)
    if len(none) == 0:
        self.exit_flag.set()
    data = self.write_socket.recv(copy=False)

    flag = True
    try:
        message, task_id, result = unpack(data)
        print(task_id)
        filename, data_id = task_id.split("#@#")
        if message == 'error':
            print(f"{data_id}预测失误，请重新预测该块数据")
            error_dict[filename] = error_dict.get(filename, 0) + 1
        else:
            write_fun(self.result_dir, filename, result)
            files_dict[filename] = files_dict.get(filename, 0) - 1
            print(f"{data_id}已写入")
    except Exception:
        self.write_socket.send(
            "failed".encode(), routing_id=data.routing_id)
        self.exit_flag.set()
        flag = False
    if flag:
        self.write_socket.send(
            "success".encode(), routing_id=data.routing_id)


class ReadWriter(Process):

    def __init__(self,
                 read_port: str,
                 write_port: str,
                 data_dir: str,
                 result_dir: str,
                 batch_size=None) -> None:
        super().__init__()
        self.read_ip = "tcp://127.0.0.1:%s" % read_port
        self.write_ip = "tcp://127.0.0.1:%s" % write_port
        self.data_dir = data_dir
        self.result_dir = result_dir
        self.batch_size = batch_size
        self.files = os.listdir(data_dir)
        self.exit_flag = Event()

    def data_genener(self):
        data_id = 0
        files = os.listdir(self.data_dir)
        data_id = 0
        for i, filename in enumerate(files):
            f = open(os.path.join(self.data_dir, filename), 'r')
            data = ''
            file_count = 0
            for start, line in enumerate(f.readlines()):
                data += line
                if self.batch_size and not (start + 1) % self.batch_size:
                    data = data
                    yield str(data_id), data, filename, -1
                    data = ''
                    file_count += 1
                    data_id += 1

            if not self.batch_size or (start + 1) % self.batch_size:
                file_count += 1
                data = data
                yield str(data_id), data, filename, file_count
                data_id += 1
                data = ''
            f.close()

    def run(self, ) -> None:
        context = zmq.Context()
        print("Connecting to  read_server(ports %s)" % self.read_ip)
        self.read_socket = context.socket(zmq.SERVER)
        self.read_socket.bind(self.read_ip)

        print("Connecting to write_server(ports %s) " % (self.write_ip))
        self.write_socket = context.socket(zmq.SERVER)
        self.write_socket.bind(self.write_ip)
        self.poller = zmq.Poller()
        self.poller.register(self.read_socket, zmq.POLLOUT | zmq.POLLIN)
        self.poller.register(self.write_socket, zmq.POLLOUT | zmq.POLLIN)
        events = self.poller.poll(2)
        print(events, zmq.POLLOUT)

        self.Q = queue.Queue()
        self.files = os.listdir(self.data_dir)
        data_iter = self.data_genener()
        files_dict = Manager().dict()
        error_dict = Manager().dict()

        async def run():
            with trio.CancelScope() as cancel_scope:
                async with trio.open_nursery() as nursery:
                    while not self.exit_flag.is_set():
                        events = self.poller.poll(2)
                        print(events, zmq.POLLOUT)
                        for socket, fd in events:
                            if fd == zmq.POLLOUT | zmq.POLLIN:
                                if socket == self.read_socket:
                                    nursery.start_soon(read_async, data_iter,
                                                       self)
                                if socket == self.write_socket:
                                    nursery.start_soon(write_async, self,
                                                       files_dict, error_dict)
                        await trio.sleep(1)
                    cancel_scope.cancel()
        trio.run(run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--read_port', type=str, default='5755')
    parser.add_argument('--write_port', type=str, default="5760")
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--data_dir',
                        type=str,
                        default="./THUCNews/testdata")
    parser.add_argument(
        '--result_dir',
        type=str,
        default="./THUCNews/resultdata")

    args = parser.parse_args()
    t = time.time()
    pi = ReadWriter(args.read_port,
                    args.write_port,
                    args.data_dir,
                    args.result_dir,
                    batch_size=args.batch_size)
    pi.start()
    pi.join()
    print(get_time_dif(t))
    for fn in os.listdir(args.result_dir):
        with open(os.path.join(args.result_dir, fn), 'r') as f:
            lines = f.readlines()
        print(fn, len(lines))
