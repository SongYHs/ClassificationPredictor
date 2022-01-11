# TODO 1 异步的调用预测服务
# TODO 2 若中断给出提示， 并且再次启动是可以从中断处开始
# 旧代码总结：
#   predictor_work_classify.py  # 启动预测客户端，其向服务端发送指令进行（读、写）服务
#   start_one.sh # 启动数据读和分发，写入服务，接受预测客户端的请求分发数据，
#   server: 4849
#   sink: 7879 : 接受写入数据，方便写入
#   dish: 4748 : 广播数据处理完毕
#   hdfs_client: 5555 ：hdfs 数据读取
#   recvData(处理数据, server)  -> request(请求新数据，server)   ->  sendPreData(写, sink) -> close(关闭)

from utils import get_time_dif, Data_Load
from predict_model import AsyncTritonModel
from config import ServerPredictConfig
from dataloader import DataGenerator
import trio
import os
import numpy as np
import time
import argparse

sep = '\t'
msg_index = 0

def get_length(datafiles):
    res ={}
    if os.path.isdir(datafiles):
        for fi in os.listdir(datafiles):
            count=0
            name = os.path.join(datafiles, fi)
            with open(name,'r') as f:
                while f.readline():
                    count+=1
            res[name] = count
    return res 


async def predict_async(filename, start,end, resultfile, server_model, data_tool, Flag,task_status=trio.TASK_STATUS_IGNORED):
        async for raw_data, data in  data_tool.read_async(filename, start, end, Flag):
            # task_status.started()
            result = server_model.predict(data)
            print(f"第{Flag}号数据已处理")
            
            data_tool.write_async(resultfile, raw_data, result)
            print(f"第{Flag}号数据已完成")



def main(files, resultdir, server_model, data_tool, batch_size):
    async def run():
        Flag = 0
        async with trio.open_nursery() as nursery:
            for i, (input_name, count) in enumerate(files.items()):
                # await trio.sleep(.1)
                for j in range(int(np.ceil(count/batch_size))):
                    start = j*batch_size 
                    Flag += 1
                    end = min([j*batch_size + batch_size, count])
                    result_name = os.path.join(resultdir,"result_"+str(i)+'.txt')
                    nursery.start_soon(predict_async, input_name, start, end , result_name, server_model, data_tool,Flag)
        
    trio.run(run)


if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='dpcnn_testtest')
    parser.add_argument('--testfile', type=str, default="THUCNews/data/test.txt")
    parser.add_argument('--dataset', type=str, default="THUCNews")
    parser.add_argument('--ngram', type=int, default=1)
    parser.add_argument('--sep', type=str, default="\t")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--pad_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--result_path', type= str, default= 'result')
    parser.add_argument('--max_size', type=int, default=50000, help = 'The number of generated sample data')
    args = parser.parse_args()
    t=time.time()

    datafiles = "./THUCNews/testdata"
    writedir = "./THUCNews/resultdata"
    config = ServerPredictConfig(args)
    vocab = DataGenerator(config).vocab
    pad = vocab.idx2word[0]
    assert 'pad' in pad.lower()
    server_model = AsyncTritonModel(config.model_name) # config.model_name
    data_tool = Data_Load(vocab, config.labels, datafiles, pad = pad)
   
    files = get_length(datafiles)
    batch_size = args.batch_size
    t=time.time()
    main(files, writedir, server_model, data_tool, batch_size)
    print(get_time_dif(t))

                


