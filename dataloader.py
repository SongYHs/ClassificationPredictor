# %%
from random import shuffle,choices
from functools import partial
from torch.utils.data import Dataset
from tqdm import tqdm
import collections
import re
import os
import pickle as pkl
import json
import torch
import numpy as np
from utils import Tokenizer


def fun_ngram(ngram, pad_size, pad,  msg):
    # msg = re.sub("[^\u4e00-\u9fa5]+",' ', msg)
    if pad_size:
        length = min([len(msg),pad_size])
    else:
        length = len(msg)
    return [msg[i:i+j+1] if i<length-j else pad for j in range(ngram) for i in range(length)]

def word_ngram(ngram, pad_size, pad):
    return partial(fun_ngram, ngram,pad_size, pad)


class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


class TextDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __getitem__(self, index):
        return torch.tensor(self.data[index][0]), torch.tensor(self.data[index][1])        

    def __len__(self):
        return(len(self.data))

    
    def aggrument(self, drop=0.0):
        pass

    def shuffle(self,d):
        np.random.permutation(d.tolist())


class DataGenerator:
    def __init__(self, config):# , data_dir,  sep='\t', ngram=1) train_ratio val_ratio test_ratio:
        self.ngram = config.ngram
        self.pad_size = config.pad_size
        self.unk, self.pad = '<unk>', '<pad>'  # 未知字，padding符号
        
        
        if not os.path.exists(config.vocab_path):
            #assert config.train_ratio+config.val_ratio+config.test_ratio == 1,f"{config.train_ratio+config.val_ratio+config.test_ratio}"
            self.train_ratio = config.train_ratio
            self.val_ratio = config.val_ratio
            self.test_ratio = config.test_ratio
            msgs, data = [], {}
            # labels =[x.strip() for x in open('THUCNews/data/class.txt', encoding='utf-8').readlines()]              # 类别名单

            with open(config.train_path, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    msg, label = line.rstrip("\n").split(config.sep)
                    # label = labels[int(label)]
                    data[label] = data.get(label, [])+[msg]
                    msgs.append(msg)
            self.vocab = self.gen_vocab(msgs)
            
            print(f"Vocab size: {len(self.vocab)}")
            self.labels = {k:i for i,k  in enumerate(data.keys())}
            
            # print(data.keys())
            # print({k:len(v) for k,v in data.items()})
            train, val, test = self.split(data)
            with open(config.train_path,'w', encoding='utf-8') as f:
                f.write("\n".join(train))
            if self.val_ratio:
                with open(config.dev_path,'w', encoding='utf-8') as f:
                    f.write("\n".join(val))
            if self.test_ratio:
                with open(config.test_path,'w', encoding='utf-8') as f:
                    f.write("\n".join(test))
            pkl.dump(self.vocab, open(config.vocab_path, 'wb'))
            json.dump(list(self.labels.keys()), open(config.label_path, "w", encoding="utf-8"))
        else:
            self.vocab = pkl.load(open(config.vocab_path, 'rb'))
            labels = json.load(open(config.label_path, 'r', encoding="utf-8"))
            self.labels = {k:i for i,k in enumerate(labels) }
        # self.tokenizer = word_ngram(self.ngram, self.pad_size, self.pad )
        self.tokenizer = Tokenizer(self.ngram, self.pad_size, self.pad)
        print(f"nums of vocab: {self.vocab.idx+1}")
        print(f"labels: {self.labels.keys()}")
        
        
    def split(self, data):
        train, val, test = [], [], []
        for label, msgs in data.items():
            msgs = [msg + "\t" + label for msg in msgs]
            shuffle(msgs)
            length = len(msgs)
            train_length, val_length = int(length*self.train_ratio), int(length*(self.train_ratio+self.val_ratio))
            train.extend(msgs[:train_length])
            val.extend(msgs[train_length:val_length])
            test.extend(msgs[val_length:])
        shuffle(train)
        shuffle(val)
        shuffle(test)
        return train, val, test
    
    def gen_vocab(self, msgs, min_freq=1, max_size=100000):
        words = [w for msg in msgs for w in self.tokenizer(msg)]
        vocabs=collections.Counter(words) 
        vocab_list = sorted([_ for _ in vocabs.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
        

        # Create a vocab wrapper and add some special tokens.
        vocab = Vocabulary()
        vocab.add_word(self.pad)
        vocab.add_word(self.unk)

        # Add the words to the vocabulary.
        for i, word in enumerate(vocab_list):
            vocab.add_word(word[0])
        
        return vocab

    def load_dataset(self, path, need_seq_len=False):
        """
           加载训练文本，并转化为数据集  
           args:  
                path: 文件路径  
           return:   
                contents: [([w1, w2, ], labeli), ]  
        """
        contents = []
        labels = {}
        with open(path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                line = line.rstrip("\n")
                if not line:
                    continue
                msg, label = line.split("\t")
                token = self.tokenizer(msg)
                tokenV = []
                for word in token:
                    tokenV.append(self.vocab(word))
                contents.append((tokenV.copy(),self.labels.get(label)))
                labels[label] = labels.get(label, 0)+1
        print(f"data distribute in  {path}: \n \t {labels}")
        return contents
    
    def load_dataset_predict(self, path, need_seq_len=False):
        """
           加载训练文本，并转化为数据集  
           args:  
                path: 文件路径  
           return:   
                contents: [([w1, w2, ], labeli), ]  
        """
        contents, labels = [], []
        labelshow = {}
        with open(path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                line = line.rstrip("\n")
                if not line:
                    continue
                msg, label = line.split("\t")
                token = self.tokenizer(msg)
                tokenV = []
                for word in token:
                    tokenV.append(self.vocab(word))
                contents.append(tokenV.copy())
                labels.append(self.labels.get(label))
                #contents.append((tokenV.copy(),self.labels.get(label)))
                labelshow[label] = labelshow.get(label, 0)+1
        print(f"data distribute in  {path}: \n \t {labelshow}")
        return contents, labels
    
    def __call__(self, train_path, dev_path, test_path):
        return TextDataset(self.load_dataset(train_path)), \
                TextDataset(self.load_dataset(dev_path)), \
                TextDataset(self.load_dataset(test_path))

                    


