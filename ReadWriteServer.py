from reader import data_genener
from multiprocessing import Process, Queue, Manager
from utils import byte2labels, read_pack, unpack, Tokenizer, Vocabulary
from writer import write_fun
import os
import zmq
import json
import argparse
from test import Timeout, timeout


@timeout(10)
def read_run(read_socket, data_iter, sep, msg_index, sequencer, data_hash):
    robot = read_socket.recv(copy=False)
    robot, routing_id = str(robot), robot.routing_id
    if robot == "request":
        try:
            data_id, data_count, raw, filename, file_count = data_iter.__next__(
            )
            data = b''
            for line in raw.split("\n"):
                msg = line.split(sep)[msg_index]
                d = '\t'.join(map(sequencer, tokenizer(msg))) + "\n"
                data += d.encode()

            send_data = read_pack("data", data_id, data_count, filename, data)
            read_socket.send(send_data, routing_id=routing_id)
            if file_count > -1:
                Q.put([filename, file_count])

            share_lock.acquire()
            data_hash[data_id] = raw  # TODO 实现data_hash的修改锁和访问锁 [x]
            share_lock.release()

        except StopIteration:
            send_data = read_pack("done", '', 0, '', b'')
            read_socket.send(send_data, routing_id=routing_id)
            return "done"
    elif robot == "exit":
        print("退出分发服务")
        return "exit"
    return "data"


def ReadWorker(read_port, data_dir, files, batch_size, tokenizer, vocab, sep,
               msg_index, Q, data_hash, share_lock):
    """
        实现数据读取
        args:
            read_port: 读取端口
            data_dir: 输入地址
            files: 输入文件列表
            batch_size: 每次预测batch_size
            tokenizer:
            sequencer:

    """
    read_port = "tcp://*:%s" % read_port
    batch_size = batch_size
    context = zmq.Context()
    print("Connecting to  read_server(ports %s)" % read_port)
    read_socket = context.socket(zmq.SERVER)
    read_socket.bind(read_port)

    data_iter = data_genener(data_dir, files, batch_size)
    sequencer = lambda token: str(vocab(token))

    while True:
        try:
            res = read_run(read_socket, data_iter, sep, msg_index, sequencer,
                           data_hash)
        except Timeout as e:
            print("读", e)
            break
        if res == "done":
            print("已读取完所有数据")
            break
        elif res == "exit":
            print("预测出现错误")
            break
    print("分发进程退出")
    read_socket.close()


# @timeout(10)
def write_run(Q, done_dict, error_dict, write_socket, data_hash, data_dir):
    if not Q.qsize() == 0:
        _data = Q.get_nowait()
        name, count = _data
        done_dict[name] = done_dict.get(name, 0) + count
    done, none, error = close_write(files, done_dict, error_dict)
    print(done, none, error)
    if len(none) == 0:
        return None, None, done, error
    data = write_socket.recv(copy=False)
    data_id, filename, result = unpack(data)
    raw = data_hash.get(data_id, None)
    if raw is not None:
        share_lock.acquire()
        data_hash.pop(data_id)  # TODO 实现data_hash的修改锁和访问锁 [x]
        share_lock.release()
        if not result[:5].tobytes().decode("utf8", "ignore") == 'error':
            result = byte2labels(result.tobytes(),
                                 labels)  # TODO 实现二进制数字向量 变为标签向量 [x]
            write_fun(data_dir, filename, raw, result)
            done_dict[filename] = done_dict.get(filename, 0) - 1
            print(f"{data_id}已写入")
        elif result[:5].tobytes().decode("utf8", "ignore") == 'error':
            error_dict[filename] = error_dict.get(filename, 0) + 1
            print(f"{data_id}预测失误，请重新预测该块数据")
    else:
        print(f"未写入数据出现丢失: {data_id}")
    write_socket.send(data_id.encode(), routing_id=data.routing_id)
    return done_dict, error_dict, done, error


def WriteWorker(write_port, data_dir, files, Q, labels, data_hash, share_lock):
    """
        实现数据写入
        args:
            write_port:
            Q: 未处理的data_id和raw
            data_dir:
    """
    write_port = "tcp://*:%s" % write_port
    context = zmq.Context()
    print("Connecting to write_server(ports %s) " % (write_port))
    write_socket = context.socket(zmq.SERVER)
    write_socket.bind(write_port)
    done_dict, error_dict = {}, {}
    while True:
        try:
            done_dict, error_dict, done, error = write_run(
                Q, done_dict, error_dict, write_socket, data_hash, data_dir)
        except Timeout as e:
            print("写", e)
            break
        if done_dict is None:
            print(f"已完成:{done}, 处理失败{error}")
            break
    print("写进程退出")
    write_socket.close()


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--read_port', type=str, default='5755')
    parser.add_argument('--write_port', type=str, default="5760")
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--data_dir',
                        type=str,
                        default="../../test/predict_client/THUCNews/testdata")
    parser.add_argument(
        '--result_dir',
        type=str,
        default="../../test/predict_client/THUCNews/resultdata")

    parser.add_argument('--ngram', type=int, default=1)
    parser.add_argument('--sep', type=str, default="\t")
    parser.add_argument('--pad_size', type=int, default=32)
    parser.add_argument('--msg_index', type=int, default=0)
    parser.add_argument(
        '--vocab_path',
        type=str,
        default='../../test/predict_client/THUCNews/data/vocab.pkl')
    parser.add_argument(
        '--label_path',
        type=str,
        default='../../test/predict_client/THUCNews/data/label.json')

    print(os.getpid())
    args = parser.parse_args()

    Q = Queue()
    share_var = Manager().dict()
    share_lock = Manager().Lock()

    vocab = json.load(open(args.vocab_path, 'r'))
    vocab = Vocabulary(word2idx=vocab["word2idx"])

    labels = json.load(open(args.label_path, 'r', encoding='utf-8'))
    tokenizer = Tokenizer(args.ngram, args.pad_size, pad_word='<pad>')

    files = os.listdir(args.data_dir)[:2]
    rp = Process(target=ReadWorker,
                 args=(args.read_port, args.data_dir, files, args.batch_size,
                       tokenizer, vocab, args.sep, args.msg_index, Q,
                       share_var, share_lock))
    rp.start()

    wp = Process(target=WriteWorker,
                 args=(args.write_port, args.result_dir, files, Q, labels,
                       share_var, share_lock))
    wp.start()

    rp.join()
    wp.join()
