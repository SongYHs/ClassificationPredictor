import re
from numpy.lib.function_base import append
from main import Classifier, get_time_dif
from config import PredictConfig
from dataloader import DataGenerator, TextDataset
from torch.utils.data import DataLoader
import argparse
import torch
import time
from predict_model import TritonModel, OnnxModel
import torch
import numpy as np
from sklearn import metrics


from utils import generator_data
import os


def evaluate(labels_all, predict_all, labels): 
    start_time = time.time() 
    acc = metrics.accuracy_score(labels_all, predict_all)
    report = metrics.classification_report(labels_all, predict_all, target_names=labels, digits=4)
    confusion = metrics.confusion_matrix(labels_all, predict_all)
    print("Precision, Recall and F1-Score...")
    print(report)
    print("Confusion Matrix...")
    print(confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    return acc, report, confusion


class Predictor:
    def __init__(self, config):
        data = DataGenerator(config)
        self.labels = config.labels
        self.batch_size=config.batch_size
        self.vocab = data.vocab.idx2word
        self.num_workers = config.num_workers
        # self.testset = TextDataset(data.load_dataset(config.testfile))
        # self.data_iter = DataLoader(self.testset, self.batch_size, num_workers =self.num_workers)
        self.inputs, self.tags = map(lambda x: np.array(x,dtype=int), data.load_dataset_predict(config.testfile))

    def predict(self, result_path, predict_model, batch_size=32):
        """
            采用predict_model 预测self.input并将结果写入result_path
        
        """

        size = self.inputs.shape[0]
        if not batch_size:
            batch_size = size
        outputs, i = np.array([], dtype=int), 0
        while i*batch_size<size:
            output = predict_model.predict(self.inputs[i*batch_size:min([i*batch_size+batch_size, size])]) # args narray ,return narray
            outputs = np.append(outputs, output)
            i+=1

        evaluate(self.tags, outputs, self.labels)
        with open(result_path, 'w', encoding='utf-8') as f:
            for i in range(size):
                msg = ''.join([self.vocab[w] for w in self.inputs[i] if w>1])
                predict = self.labels[outputs[i]]
                label = self.labels[self.tags[i]]
                f.write('\t'.join([msg, predict, label]))
                f.write('\n')

    def predict_iter(self, result_path, predict_model, batch_size=32, max_size = 100000):
        """
            采用predict_model 预测self.input并将结果写入result_path
        
        """
        data_iters = generator_data(self.inputs, self.tags, batch_size, max_size)
        
        f =  open(result_path, 'w', encoding='utf-8')
        metric_acc, length = 0.0 , 0 
        for data, tag in data_iters.__iter__():
            batch = tag.shape[0]
            # print(batch, len(tag),data.shape, tag)
            output = predict_model.predict(data)
            acc = metrics.accuracy_score(tag, output)
            metric_acc += batch*acc
            length += batch
            
            for i in range(batch):
                msg = ''.join([self.vocab[w] for w in data[i] if w>1])
                predict = self.labels[output[i]]
                label = self.labels[tag[i]]
                f.write('\t'.join([msg, predict, label]))
                f.write('\n')
        f.close()

        print(f"预测精度:{metric_acc/length}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_onnx', type=int, default=0)
    # parser.add_argument('--predict_model', type=str, default='torch', choices=["torch",'onnx', 'triton'])
    parser.add_argument('--predict_model', type=str, default='triton', choices=["torch",'onnx', 'triton'])
    parser.add_argument('--testfile', type=str, default="data/test.txt")
    parser.add_argument('--dataset', type=str, default="THUCNews")
    parser.add_argument('--model_path', type=str, default="THUCNews/saved_dict/DPCNN.ckpt")
    parser.add_argument('--ngram', type=int, default=1)
    parser.add_argument('--sep', type=str, default="\t")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--pad_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--iters', type=int, default=3)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--result_path', type= str, default= 'result')
    parser.add_argument('--max_size', type=int, default=50000, help = 'The number of generated sample data')
    args = parser.parse_args()
    t=time.time()
    config = PredictConfig(args)
    print(f"加载参数 time={get_time_dif(t)}")


    predictor = Predictor(config)
    if args.predict_model == 'torch':
        config.embedding_pretrained = torch.load(args.model_path)['embedding.weight']
        config.word_embedding_dimension = config.embedding_pretrained.size(1)
        config.word_num = config.embedding_pretrained.size(0)
        model = Classifier(config)
        model.model.load(config.model_path)
    elif args.predict_model == 'onnx':
        model = OnnxModel(args.model_path)
    elif args.predict_model == 'triton':
        model = TritonModel()      
    sec=[]

    result_file = os.path.join(args.result_path, 'result_' + args.predict_model+".txt")

    for i in range(args.iters):
        t0=time.time()
        # predictor.predict('data/'+args.predict_model+'.txt', model)
        predictor.predict_iter(result_file, model, args.batch_size, args.max_size)
        t = get_time_dif(t0)
        print(f"预测_tritron time={t}")
        sec.append(t.seconds)
    print(sum(sec)/args.iters)

