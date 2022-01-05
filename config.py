import numpy as np
import torch
import json

class Config(object):

    """配置参数"""
    def __init__(self, dataset, embedding):
        self.model_name = 'DPCNN'
        self.train_path = dataset + '/data/train.txt'                                # 训练集
        self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        self.test_path = dataset + '/data/test.txt'                                  # 测试集
        self.label_path = dataset+ '/data/label.json'
        self.vocab_path = dataset + '/data/vocab_pre.pkl'                                # 词表
        self.save_path = dataset + '/saved_dict/' + self.model_name + '_new.ckpt'        # 模型训练结果
        self.export_onnx_file = dataset +"/saved_dict/" + self.model_name +"_new.onnx"
        self.log_path = dataset + '/log/' + self.model_name
        self.labels = json.load(open(self.label_path,'r',encoding='utf-8'))
        self.num_classes=len(self.labels)
        self.is_embedding = True
        self.embedding_pretrained = torch.tensor(
            np.load(dataset + '/data/' + embedding)["embeddings"].astype('float32'))\
            if embedding != 'random' else None                                       # 预训练词向量
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.ngram = 1
        self.train_ratio = 1.0
        self.val_ratio = 0.0
        self.test_ratio = 0.0
        self.sep="\t"

        self.dropout = 0.5                                              # 随机失活
        self.require_improvement = 100                                 # 若超过1000batch效果还没提升，则提前结束训练
        
        self.word_num = self.embedding_pretrained.size(0)\
            if self.embedding_pretrained is not None else 5000           # 字向量维度                                               # 词表大小，在运行时赋值
        self.num_epochs = 50                                         # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3                                       # 学习率
        self.word_embedding_dimension = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度
        self.channel_size = 250                                          # 卷积核数量(channels数)
        self.num_workes=4

class PredictConfig(object):

    """配置参数"""
    def __init__(self, args):
        self.model_name = 'DPCNN'
        self.learning_rate = 1e-3 
        self.is_onnx = args.is_onnx
        self.testfile = args.testfile
        self.label_path = args.dataset + '/data/label.json'
        self.vocab_path = args.dataset + '/data/vocab_pre.pkl'                                # 词表
        self.model_path = args.model_path
        

        self.labels = json.load(open(self.label_path,'r',encoding='utf-8'))
        self.num_classes=len(self.labels)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.ngram = args.ngram
        self.sep = args.sep

        self.batch_size = args.batch_size # 64                                           # mini-batch大小
        self.pad_size = args.pad_size # 32                                              # 每句话处理成的长度(短填长切)
        self.num_workers= args.num_workers # 4
        self.is_embedding = True
        self.channel_size = 250                                          # 卷积核数量(channels数)
        self.word_num = 3896
        self.word_embedding_dimension =200
        self.embedding_pretrained = None