import numpy as np
import re
import os
import trio 
import random
import time

from datetime import timedelta

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

class Tokenizer(object):
    def __init__(self, ngram, pad_size, pad_word, vocab=None,  stop_word=None, only_chinese=False):
        self.stop_word = stop_word
        self.ngram = ngram
        self.pad_size = pad_size
        self.pad_word = pad_word
        self.only_chinese = only_chinese
        if vocab:
            self.trie= Trie(vocab)
        else:
            self.trie = None
    
    def clear_word(self, msg):
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


async def aenumerate(ait, start=0):
        i = start
        async for item in ait:
            yield i, item
            i += 1

def generator_data(inputs, tags, batch_size, max_size, flag=False):
    size0 = inputs.shape[0]
    ii=0

    if not flag:
        while ii<max_size:
            # for j in range(int(10000000*np.random.randint(1,10))):
            #     pass
            # print(ii)
            start = ii % size0
            now = min([(ii+batch_size), max_size])  # 50000
            end =  now % size0 
            # print(start, end, now)

            if not int(ii / size0) == int(now / size0):
                data = np.append(inputs[start:], inputs[:end], axis=0) #[i*max_len:min([i*max_len+max_len, size])] 
                tag = np.append(tags[start:], tags[:end]) #[i*max_len:min([i*max_len+max_len, size])] 
            else:
                data = inputs[start:end]
                tag = tags[start:end]
            ii += batch_size
            yield data, tag
    else:
        while ii<max_size:
            start = ii
            end = min([ii+batch_size, max_size])
            ii += batch_size
            yield inputs[start: end], tags[start: end]


class Data_Load:
    def __init__(self, vocab, labels, datafiles, ngram=1, pad_size=32, pad ='<pad>',sep = '\t', msg_index=0):
        self.vocab = vocab
        self.labels = labels
        self.tokenizer =  Tokenizer(ngram, pad_size, pad)
        self.sep = sep
        self.msg_index = msg_index
        if os.path.isdir(datafiles):
            self.datafiles = [os.path.join(datafiles, d) for d in os.listdir(datafiles)]
    
    def read_data(self, batch_size):
        """
            从多个文件下串行读取训练数据
            args:

        """
        while True:
            data =[]
            fn = self.datafiles.pop()
            f = open(fn, encoding = 'utf-8')
            for i in range(batch_size):
                line = f.readline()
                if line:
                    data.append(line)
                else:
                    f.close()
                    if not self.datafiles: break
                    fn = self.datafiles.pop()
                    f = open(fn, encoding = 'utf-8')
            yield self.token_vocab(data)

    # async def Aread_data(self, batch_size):
    #     """
    #         实现异步读取多个文件： 
    #             方式1：将所有文件排序作为一个长文件读取，异步读一个文件
    #             方式2：每步读一个文件，同一文件内阻塞读取
    #     """
    async def read_async(self, filename, start, end, ii ): 
        """ 读取filename的start到end行 
        """
        async with await trio.open_file(filename) as f:
            # data = np.array([], dtype=int)
            t=time.time()
            raw, data = [], []
            async for i, line in aenumerate(f):
                if i >= start and i<end:
                    msg = line.rstrip('\n').split(self.sep)[self.msg_index]
                    raw.append(msg)
                    # ms = np.array(self.token_vocab(msg), dtype=int)
                    # data = np.append(data, ms)
                    data.append(self.token_vocab(msg))
                elif i==end:
                    
                    data = np.array(data,dtype=int)
                    # print(data.shape)
                    print(f"已读取第{ii}号数据，共{end-start}条数据,耗时{get_time_dif(t)}")
                    yield raw,data
                    break

    async def read_async_2(self, filename, count):
        """
        """
        pass


    async def write_async(self, filename, raw, output):
        async with await trio.open_file(filename, mode='w') as f:
            for line, out in zip(raw, output):
                tag = self.labels[out]
                f.write(self.sep.join([line, tag]))


    
    def token_vocab(self, msg):
        """
            数据序列化
        """
        res = [self.vocab(token) for token in self.tokenizer(msg)]
        # print(res,len(res))
        return res



# %% 
def test_async_datas():
    t=time.time()
    
    async def write_async(ii):
        async for i in read_async(ii):
            print(i)

    async def read_async(ii ): 
        async with await trio.open_file("./THUCNews/data/test.txt") as f:
            async for i, line in aenumerate(f):
                if i==ii:
                    await trio.sleep(1)
                    # print(i,line)
                    yield line
                    break

    async def run():
        async with trio.open_nursery() as nursery:
            for ii in range(10):
                # await trio.sleep(0.1)
                nursery.start_soon(write_async, ii)
    trio.run(run)
    print("耗时: ",get_time_dif(t))

# %%