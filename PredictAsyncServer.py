
from predictor import TritonModel, InferenceServerException
from multiprocessing import Process, Queue, Manager
from utils import byte2narray, get_time_dif, pack, read_unpack
import numpy as np
import queue
import time
import logging
import os
import argparse
import zmq
import trio


def exit_read(read_data, process_num):
    for _ in range(process_num):
        read_data.put(None)


def ReadQueue(read_port, read_data, process_num):
    """
        实现远程传输的数据入队列
        args:
            read_port: 传输接口
            read_data: 输入数据队列
            process_num: 预计启动的预测处理的进程个数
    """
    print(f"输入服务连接{read_port}")
    context = zmq.Context()
    read_socket = context.socket(zmq.CLIENT)
    read_socket.connect(read_port)
    while True:
        if not read_data.full():
            read_socket.send("request".encode())
            d = read_socket.recv()
            msg, data_id, data_count, filename, raw = read_unpack(d)
            if msg == "done":
                exit_read(read_data, process_num)
                print("已读取所有数据, 等待处理结束")
                break
            elif msg == "data":
                data = byte2narray(raw)  # TODO 二进制数据转为numpy.narray
                read_data.put([data_id, data, filename])
                # print(f"已读取第{data_id}号数据，共{data_count}条数据")


def PredictWorker(read_data, model_name, enterpoint, is_grpc, write_data,
                  retry_time):
    """
        实现预测处理任务
        args:
            read_data: 输入数据队列
            model_name: 启动的triton模型名称
            enterpoint: triton 服务的端口
            is_grpc: 是否采用grpc和triton服务通信
            write_data: 处理后数据的队列
            retry_time: 模型预测的重试次数

    """
    server_model = TritonModel(
        model_name, enterpoint,
        is_grpc)  # model_name, predict_enterpoint, is_grpc) # config.model_name

    share_var = Manager().dict()
    share_lock = Manager().Lock()

    async def run():
        with trio.CancelScope() as cancel_scope:
            async with trio.open_nursery() as nursery:
                nursery.start_soon(predict, read_data, share_var, server_model,
                                   share_lock)
                nursery.start_soon(write_results, share_var, write_data,
                                   retry_time, share_lock)

    trio.run(run)


async def predict(read_data, results, server_model, share_lock):
    while True:
        if not read_data.qsize() == 0:
            try:
                _data = read_data.get_nowait()
                if _data == None:
                    print(os.getpid(), "退出")
                    break  # 接收到None信号结束进程
                data_id = _data[0]
                data = _data[1]
                filename = _data[2]
                del _data
            except queue.Empty:
                data_id = False
            if data_id is not False:
                if len(data) > 0:
                    request_id = filename + '#@#' + str(data_id)
                    server_model.predict_grpc_async(data, results, request_id,
                                                    share_lock)
                    # print(f"开始预测{request_id}")


async def write_results(results, write_data, retry_time, share_lock):
    retry, timeout_retry = 0, 0

    while True:
        # if lock.value==3:
        if True:
            if not len(results) == 0:
                try:
                    await share_lock.acquire()
                    _data = results.pop()
                    share_lock.release()
                    if _data == None:
                        break
                    if retry >= retry_time:
                        print(f"预测{retry_time}次错误,退出预测")
                        write_data.put([data_id, "error", filename])
                        break
                    if type(_data) == InferenceServerException:
                        print(f"预测出现{_data}错误，重试{retry}次")
                        retry += 1
                        continue
                    retry, timeout_retry = 0, 0
                    filename, data_id = map(
                        str,
                        _data.get_response().id.split('#@#'))
                    result = np.argmax(_data[0].as_numpy('output'), 1)
                    write_data.put([data_id, result, filename])
                except queue.Empty:
                    print(f"预测超时{timeout_retry}次")
                    timeout_retry += 1
                if timeout_retry > 5:
                    print(f"预测超时{timeout_retry}次，退出")
                    write_data.put([data_id, "error", filename])
                    break
            else:
                pass
                # await trio.sleep(1)
                # print(f"等待数据预测完成")


def WriteQueue(write_port, write_data):
    """
        实现处理后的数据远程传输
        args:
            write_port: 传输接口
            write_data: 处理后数据的队列

    """
    print(f"输出服务连接{write_port}")
    context = zmq.Context()
    write_socket = context.socket(zmq.CLIENT)
    write_socket.connect(write_port)

    while True:
        if write_data.full():
            print("后处理太慢，已阻塞")
        if not write_data.qsize() == 0:
            try:
                _data = write_data.get_nowait()
                if _data == None:
                    print("后处理退出")
                    break
                data_id = _data[0]
                result = _data[1]
                filename = _data[2]
            except queue.Empty:
                data_id = "False"
            if not data_id == "False":
                # print(raw)
                result = '\n'.join(map(
                    str, result)).encode() if not isinstance(
                        result,
                        str) else "error".encode()  # TODO narray 转byte [x]
                send_data = pack(data_id, filename, result)
                write_socket.send(send_data)
                result = write_socket.recv()
                print(f"{result}数据已写入")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='dpcnn_onnx')
    parser.add_argument('--read_port', type=str, default='5755')
    parser.add_argument('--write_port', type=str, default="5760")
    parser.add_argument('--predict_enterpoint',
                        type=str,
                        default='0.0.0.0:8001')
    parser.add_argument('--is_grpc', type=int, default=1)
    parser.add_argument('--process_num', type=int, default=1)
    args = parser.parse_args()
    t = time.time()
    log = logging.getLogger('Classify_Predict_Worker: ')
    log.setLevel(logging.DEBUG)

    read_port = "tcp://127.0.0.1:%s" % args.read_port
    write_port = "tcp://127.0.0.1:%s" % args.write_port
    read_data = Queue(10)
    write_data = Queue(10)

    t = time.time()
    Ar = Process(target=ReadQueue,
                 args=(read_port, read_data, args.process_num))
    Aw = Process(target=WriteQueue, args=(write_port, write_data))
    Ar.start()
    Aw.start()
    Ap = []

    for p in range(args.process_num):
        Ap.append(
            Process(target=PredictWorker,
                    args=(read_data, args.model_name, args.predict_enterpoint,
                          args.is_grpc, write_data, 4)))
        Ap[-1].start()
    for p in Ap:
        p.join()
    write_data.put(None)
    Aw.join()
    print(f"耗时: {get_time_dif(t)}s")
