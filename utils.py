import re
import time
import numpy as np
from datetime import timedelta

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def pack(msg, data_id, data):
    msg_b = msg.encode()
    data_id_b = data_id.encode()
    msg_len = len(msg_b)
    data_id_len = len(data_id_b)
    new_data = chr(msg_len).encode() + chr(data_id_len).encode() + msg_b + data_id_b + data
    return new_data


def unpack(data):
    data = memoryview(data)
    msg_len = data[0]
    data_id_len = data[1]
    msg = data[2:msg_len + 2].tobytes().decode()
    data_id = data[msg_len+2:data_id_len+msg_len+2].tobytes().decode()
    new_data = data[msg_len+data_id_len + 2:]
    return msg, data_id, new_data



def read_pack(msg, data_id, data_count, output_name, data):
    msg_b = msg.encode()
    data_id_b = data_id.encode()
    data_count_b = str(data_count).encode()
    output_name_b = output_name.encode()
    msg_len = len(msg_b)
    data_id_len = len(data_id_b)
    count_len = len(data_count_b)
    out_len = len(output_name_b)
    new_data = chr(msg_len).encode() +  \
                chr(data_id_len).encode()+ \
                chr(count_len).encode()+ \
                chr(out_len).encode()+ \
                msg_b + data_id_b + data_count_b + output_name_b +  data
    return new_data

def read_unpack(data):
    data = memoryview(data)
    msg_len = data[0]
    id_len = data[1]
    count_len = data[2]
    output_len = data[3]
    msg = data[4:msg_len + 4].tobytes().decode()
    data_id = data[msg_len+4:id_len+msg_len+4].tobytes().decode()
    data_count = int(data[msg_len+id_len+4: msg_len+id_len+count_len+4].tobytes().decode())
    output_name = data[msg_len+id_len+count_len+4 :msg_len+id_len+count_len+output_len+4].tobytes().decode()
    new_data = data[msg_len+id_len+count_len+output_len+4:-1]
    return msg, data_id, data_count, output_name, new_data


def byte2narray(data):
    return np.array([[int(i) for i in l.split(b'\t')] for l in data.tobytes().split(b'\n') ])

def byte2labels(data, labels):
    fun = lambda x:labels[int(x)] 
    return map(fun, data.split(b"\n"))


class Tokenizer(object):
    def __init__(self, ngram, pad_size, pad_word, vocab=None,  stop_word=None, only_chinese=False):
        self.ngram = ngram
        self.pad_size = pad_size
        self.pad_word = pad_word
        self.stop_word = stop_word
        self.only_chinese = only_chinese
        if vocab:
            self.trie= Trie(vocab)
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
            msg = re.sub(stop_word,'', msg)
        if self.only_chinese:
            msg = re.sub("[^\u4e00-\u9fa5]+",' ', msg)
        return msg
    
    def __call__(self, msg):
        msg = self.clear_word(msg)
        if not self.trie:
            length = len(msg)
            return [msg[i:i+j+1] if i<length-j else self.pad_word for j in range(self.ngram) for i in range(self.pad_size)]
        else:
            return self.trie.tokenizer(msg)


class Trie(object):
    def __init__(self, vocab):
        self.root = None
        for k,_ in vocab.items():
            self.add(k)

    def add(self, word):
        pass

    def tokenizer(self, msg):
        pass

class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self, word2idx=dict(), idx2word= dict(), unk = '<unk>', pad='<pad>'):
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
