import pytest
import numpy as np
import pickle as pkl
from multiprocessing import Process, Lock, Event, Pool, Queue
import queue
import time
import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException
import logging
import os, sys
import argparse
import uuid
from functools import partial
import zmq
import json
from utils import pack, unpack
import re
from utils import get_time_dif


class Tokenizer(object):

    def __init__(self,
                 ngram,
                 pad_size,
                 pad_word,
                 vocab=None,
                 stop_word=None,
                 only_chinese=False):
        self.ngram = ngram
        self.pad_size = pad_size
        self.pad_word = pad_word
        self.stop_word = stop_word
        self.only_chinese = only_chinese
        if vocab:
            self.trie = Trie(vocab)
        else:
            self.trie = None

    def clear_word(self, msg):
        """
            清洗msg:
                如果stop_word非空，去除停用词
                如果only_chinese非空，去除标点符号和英文字符
        """
        if self.stop_word:
            stop_word = f"[^{'|'.join(self.stop_word)}]+"
            msg = re.sub(stop_word, '', msg)
        if self.only_chinese:
            msg = re.sub("[^\u4e00-\u9fa5]+", ' ', msg)
        return msg

    def __call__(self, msg):
        msg = self.clear_word(msg)
        if not self.trie:
            length = len(msg)
            return [
                msg[i:i + j + 1] if i < length - j else self.pad_word
                for j in range(self.ngram) for i in range(self.pad_size)
            ]
        else:
            return self.trie.tokenizer(msg)


class Trie(object):

    def __init__(self, vocab):
        self.root = None
        for k, _ in vocab.items():
            self.add(k)

    def add(self, word):
        pass

    def tokenizer(self, msg):
        pass


class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self,
                 word2idx=dict(),
                 idx2word=dict(),
                 unk='<unk>',
                 pad='<pad>'):
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.idx = len(word2idx)
        self.unk = unk
        self.pad = pad

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx[self.unk]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def _callback(user_data, result, error):
    if error:
        user_data.append(error)
    else:
        user_data.append(result)


class TritonModel:

    def __init__(self,
                 model_name='dpcnn_testtest',
                 enterpoint='10.1.60.158:8000',
                 is_grpc=True):
        if is_grpc:
            self.triton_client = grpcclient.InferenceServerClient(
                url=enterpoint)
            self.predict = self.predict_grpc
        else:
            self.triton_client = httpclient.InferenceServerClient(
                url=enterpoint)
            self.predict = self.predict_http
        self.model_name = model_name

    def predict_http(self, data):
        """
            args:
                dataset: numpy.ndarray, [batch_size, pad_size]
            return:   
                output: numpy.ndarray, [batch_size]
        """
        inputs = []
        inputs.append(httpclient.InferInput('input', data.shape, "INT64"))
        # inputs[0].set_data_from_numpy(data)
        inputs[0].set_data_from_numpy(data, binary_data=False)
        outputs = []
        outputs.append(
            httpclient.InferRequestedOutput('output',
                                            binary_data=False))  # 获取 1000 维的向量
        # results = self.triton_client.infer(self.model_name, inputs=inputs, outputs=outputs)

        # return np.argmax(results.as_numpy('output'), 1)
        results = self.triton_client.async_infer(self.model_name,
                                                 inputs=inputs,
                                                 outputs=outputs)

        return np.argmax(results.get_result().as_numpy('output'), 1)

    def predict_grpc(self, data):
        """
            args:
                dataset: numpy.ndarray, [batch_size, pad_size]
            return:   
                output: numpy.ndarray, [batch_size]
        """
        inputs = []
        inputs.append(grpcclient.InferInput('input', data.shape, "INT64"))
        # inputs[0].set_data_from_numpy(data)
        inputs[0].set_data_from_numpy(data)
        outputs = []
        outputs.append(
            grpcclient.InferRequestedOutput('output'))  # 获取 1000 维的向量

        user_data = []
        self.triton_client.async_infer(self.model_name,
                                       inputs=inputs,
                                       outputs=outputs,
                                       callback=partial(_callback, user_data))
        time_out = 10
        while ((len(user_data) == 0) and time_out > 0):
            time_out = time_out - 1
            time.sleep(2)
        if ((len(user_data) == 1)):
            # Check for the errors
            if type(user_data[0]) == InferenceServerException:
                print(user_data[0])
                sys.exit(1)

        return np.argmax(user_data[0].as_numpy('output'), 1)


def exit_read(read_data, process_num):
    for _ in range(process_num):
        read_data.put(None)


def ReadWork(read_port, read_data, sep, msg_index, tokenizer, vocab,
             process_num):
    print(f"输入服务连接{read_port}")
    context = zmq.Context()
    read_socket = context.socket(zmq.CLIENT)
    read_socket.connect(read_port)
    while True:
        if not read_data.full():
            read_socket.send("request".encode())
            d = read_socket.recv()
            msg, task_id, raw = unpack(d)
            raw = raw.tobytes().decode()
            print(task_id, "**", msg)

            if msg == "done":
                exit_read(read_data, process_num)
                print(f"已读取所有数据, 等待处理结束")
                break
            elif msg == "data":
                filename, data_id = task_id.split("#@#")
                d = []
                for line in raw.split("\n"):
                    msg = line.split(sep)[msg_index]
                    d.append([vocab(token) for token in tokenizer(msg)])
                d = np.array(d, dtype=int)
                read_data.put([data_id, d, raw, filename])
                print(f"已读取第{data_id}号数据，共{d.shape}条数据, {read_data.qsize()}")


def PredictWorker(read_data, model_name, predict_enterpoint, is_grpc,
                  write_data, retry_time):
    server_model = TritonModel(model_name, predict_enterpoint,
                               is_grpc)  # config.model_name
    while True:
        time.sleep(1)
        if not read_data.qsize() == 0:
            try:
                _data = read_data.get_nowait()
                if _data == None:
                    print(os.getpid(), "退出")
                    break  # 接收到None信号结束进程
                data_id = _data[0]
                data = _data[1]
                raw = _data[2]
                resultfile = _data[3]
                del _data
            except queue.Empty:
                data_id = False
            if data_id is not False:
                if len(data) > 0:
                    retry = 0
                    while True:
                        try:
                            result = server_model.predict(data)
                            write_data.put([data_id, result, raw, resultfile])
                            break
                        except:
                            retry += 1
                            print(f"{data_id}号数据，模型预测错误，第{retry}次重试中")
                        if retry >= retry_time:
                            print(f"{data_id}号数据，{retry}次重试错误")
                            write_data.put([data_id, "error", "", resultfile])
                            break

        else:
            print(
                f"{os.getpid()}等待读取数据{read_data.qsize()}AND{read_data.empty()},{os.getpid()}"
            )
            time.sleep(1)


def WriteWork(write_port, write_data, labels):
    print(f"输出服务连接{write_port}")
    context = zmq.Context()
    write_socket = context.socket(zmq.CLIENT)
    write_socket.connect(write_port)
    funs = lambda x: x[0] + "\t" + labels[x[1]]
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
                raw = _data[2]
                filename = _data[3]
            except queue.Empty:
                data_id = "False"
            if not data_id == "False":
                # print(raw)
                if isinstance(result, str):
                    flag = "error"
                    data = result
                else:
                    flag = "result"
                    data = "\n".join(map(funs, zip(raw.split("\n"), result)))
                task_id = filename + "#@#" + str(data_id)
                send_data = pack(flag, task_id, data)
                write_socket.send(send_data)
                result = write_socket.recv()
                print(f"{result}数据已写入")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='dpcnn_onnx')
    parser.add_argument('--read_port', type=str, default='5755')
    parser.add_argument('--write_port', type=str, default="5760")
    parser.add_argument('--ngram', type=int, default=1)
    parser.add_argument('--sep', type=str, default="\t")
    parser.add_argument('--pad_size', type=int, default=32)
    parser.add_argument(
        '--vocab_path',
        type=str,
        default='../../test/predict_client/THUCNews/data/vocab.json')
    parser.add_argument(
        '--label_path',
        type=str,
        default='../../test/predict_client/THUCNews/data/label.json')
    parser.add_argument('--msg_index', type=int, default=0)
    parser.add_argument('--predict_enterpoint',
                        type=str,
                        default='0.0.0.0:8001')
    parser.add_argument('--is_grpc', type=int, default=1)
    parser.add_argument('--process_num', type=int, default=1)
    args = parser.parse_args()
    t = time.time()
    log = logging.getLogger('Classify_Predict_Worker: ')
    log.setLevel(logging.DEBUG)
    _id = str(uuid.uuid4())

    vocab = json.load(open(args.vocab_path, 'r'))
    vocab = Vocabulary(word2idx=vocab["word2idx"])
    labels = json.load(open(args.label_path, 'r', encoding='utf-8'))
    tokenizer = Tokenizer(args.ngram, args.pad_size, pad_word='<pad>')
    read_port = "tcp://127.0.0.1:%s" % args.read_port
    write_port = "tcp://127.0.0.1:%s" % args.write_port
    read_data = Queue(10)
    write_data = Queue(10)

    t = time.time()
    Ar = Process(target=ReadWork,
                 args=(read_port, read_data, args.sep, args.msg_index,
                       tokenizer, vocab, args.process_num))
    Aw = Process(target=WriteWork, args=(write_port, write_data, labels))
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
