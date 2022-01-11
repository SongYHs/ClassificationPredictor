# TODO 定义Classifier, 实现训练, 预测, 验证步骤

import argparse
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from config import Config
from dataloader import DataGenerator, TextDataset
from sklearn import metrics
from importlib import import_module
from model import DPCNN
import time
import os
from utils import get_time_dif


# parser = argparse.ArgumentParser()
# parser.add_argument('--lr', type=float, default=0.1)
# parser.add_argument('--batch_size', type=int, default=16)
# parser.add_argument('--epoch', type=int, default=20)
# parser.add_argument('--gpu', type=int, default=0)
# parser.add_argument('--out_channel', type=int, default=2)
# parser.add_argument('--label_num', type=int, default=2)
# parser.add_argument('--seed', type=int, default=1)
# args = parser.parse_args()

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# torch.manual_seed(args.seed)

if torch.cuda.is_available():
    torch.cuda.set_device(0)

# Create the configuration
# config = Config(sentence_max_size=50,
#                 batch_size=args.batch_size,
#                 word_num=11000,
#                 label_num=args.label_num,
#                 learning_rate=args.lr,
#                 cuda=args.gpu,
#                 epoch=args.epoch,
#                 out_channel=args.out_channel)
MODEL = {
    "DPCNN": DPCNN
}



class Classifier:
    def __init__(self, config):
        self.config = config
        assert config.model_name in MODEL.keys(), f"模型加载失败，无{config.model_name}模型"
        self.model = MODEL[config.model_name](config)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=config.learning_rate)
        if torch.cuda.is_available():
            self.model.cuda()
        self.max_f1score = 0.0
    
    def train(self, train_set, val_set= None):
        
        trainset = DataLoader(train_set, self.config.batch_size,shuffle=True )
        for epoch in range(self.config.num_epochs):
            self.model.train()
            t=time.time()
            loss_all = 0
            for i, (data, label) in enumerate(trainset):
                data = data.to(self.config.device)
                label = label.to(self.config.device)
                out = self.model(data)
                
                self.optimizer.zero_grad()
                loss = self.criterion(out, label)
                loss.backward()
                self.optimizer.step()
                loss_all += loss.item()
                if not (i+1)%100:
                    true = label.data.cpu()
                    predic = torch.max(out.data, 1)[1].cpu()
                    train_acc = metrics.accuracy_score(true, predic)
                    time_dif = get_time_dif(t)
                    msg = 'Epoch:{4}, Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%} Time: {3}'
                    print(msg.format(i+1, loss.item(), train_acc, time_dif, epoch))

            # print(f"time = {get_time_dif(t)}, epoch = {epoch}, loss = {loss_all/len(trainset)}")
            if val_set:
                _, _, _, _, f = self.eval(val_set)
                if f> self.max_f1score:
                    self.model.save(self.config.save_path)
                    self.max_f1score = f
                    print(f"save_model in {self.config.save_path}")
                    
            print(f"time = {get_time_dif(t)}, epoch = {epoch}, loss = {loss_all/len(trainset)}")
    
    def eval(self, val_set):
        self.model.eval()
        valset = DataLoader(val_set, self.config.batch_size, num_workers=4)
        loss_total = 0
        predicts_all = np.array([], dtype=int)
        labels_all = np.array([], dtype=int)
        for data, label in valset:
            data = data.to(self.config.device)
            label = label.to(self.config.device)
            output = self.model(data)
            loss = F.cross_entropy(output, label)
            loss_total += loss
            label = label.data.cpu().numpy()
            predicts = torch.max(output.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, label)
            predicts_all = np.append(predicts_all, predicts) 
        acc = metrics.accuracy_score(labels_all, predicts_all)
        p, r, f, _ = metrics.precision_recall_fscore_support(labels_all, predicts_all,average='macro')
        # 打印各类别的PRF
        print("\t\t  P  \t  R  \t  F  \t  NumLabel", )
        for i, label in enumerate(self.config.labels):
            # predicts = [p==i for p in predicts_all]
            num = sum([l==i for l in labels_all])
            lp, lr, lf, _ = metrics.precision_recall_fscore_support(labels_all, predicts_all, average='macro', labels=[i])
            print("%s \t %.4f \t %.4f \t %.4f %d" % (label, lp, lr, lf, num))
        print("%s \t %.4f \t %.4f \t %.4f" %("macro-ALL", p, r, f))
        return acc, loss_total/len(valset), p, r, f
    
    def predict(self, dataset):
        """
            args:
                dataset: numpy.ndarray, [batch_size, pad_size]
            return:
                output: numpy.ndarray, [batch_size]

        """
        self.model.eval()
        input = torch.tensor(dataset).to(self.config.device)
        output = self.model(input)
        return torch.max(output.data, 1)[1].cpu().numpy()


    def save_onnx(self):
        device = self.config.device
        input_shape = (64, self.config.pad_size)   #输入数据,改成自己的输入shape
        x = torch.randint(self.config.word_num, input_shape).to(device)
        example_outputs=torch.rand((64,10)).to(device)
        # print(example_outputs, example_outputs.shape)
        self.model.eval()
        print(self.config.export_onnx_file)
        torch.onnx.export(self.model,
                    x,
                    self.config.export_onnx_file,
                    opset_version=10,
                    do_constant_folding=True,	# 是否执行常量折叠优化
                    input_names=["input"],	# 输入名
                    output_names=["output"],
                    example_outputs = example_outputs
                    ,	# 输出名
                    dynamic_axes={"input":{0:"batch_size"},  # 批处理变量
                                    "output":{0:"batch_size"}})

    


if __name__ == "__main__":
    
    # embedding = args.embedding
    # data_name = args.data_name
    embedding = 'embedding_NewSougou.npz'
    data_name = 'THUCNews'
    config = Config(data_name, embedding)
    print(config.embedding_pretrained.shape)
    save_dir, filename = os.path.split(config.save_path)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        print(f"文件保存于{save_dir}")

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    # 加载数据
    dataset = DataGenerator(config)
    config.word_num = len(dataset.vocab.idx2word)
    train_set, dev_set, test_set = dataset(config.train_path, config.dev_path, config.test_path)
    
    # 加载模型
    model = Classifier(config) 
    if True:
        model.model.load(config.save_path)
    # print("模型开始训练...")
    # model.train(train_set, dev_set)
    # # model.train(train_set)
    # model.eval(dev_set)

    print("模型保存")
    model.save_onnx()




