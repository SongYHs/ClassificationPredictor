from predict import *
from  multiprocessing import Process,Lock
import zmq
from utils import generator_data
# def distribute_data(datafile, offset, max_len):
#     """
#         分发数据
#         args:
#             datafile: 数据源
#             offset: 起始行
#             max_len: 最多读取行数
#     """
#     pass
Client_FLAG = False
metric_acc, length =0,0

def write_data(config, input, tag, output,  mlock,f  ):
    """
        写入数据
    """
    mlock.acquire()
    size = input.shape[0]
    f.writelines()
    for i in range(size):
        msg = ''.join([config.vocab[w] for w in input[i] if w>1])
        predict = config.labels[output[i]]
        label = config.labels[tag[i]]
        f.write('\t'.join([msg, predict, label]))
        f.write('\n')

    mlock.release()

def server(port,config,  server_model, data_iters, server_count, mlock=None,f=None):
    """
        处理进程, 包括读取数据，和服务器沟通，及写入
        args:
            port:  进程通信端口号
            server_model: 服务端，其包含predict方法，对输入矩阵进行预测
            config: 包括vocab, labels, datafile
            data_iters: 数据生成器
            server_count: 各服务与预测服务通信次数
    """
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:%s" % port)
    print ("Running server on port: ", port)

    for i in range(server_count):
        message = socket.recv()
        i = int(message)
        
        data,tag= data_iters.__next__()
        # print(f"{port}第{i}号数据加载完成{tag.shape[0]}")
        socket.send(str(tag.shape[0]).encode("ascii") )
        output = server_model.predict(data)
        acc = metrics.accuracy_score(tag, output)

        # 模拟定时
        # for i in range(int(100000000*np.random.randint(1,10))):
        #     pass
        write_data(config, data, tag, output, mlock,f )
        print(f"{port}第{i}号数据写入{tag.shape[0]}完成")
        
        
           

def client(ports=["5556"],  c=10):
    """
        与处理进程进行通信，分发数据
        args:
            ports: 进程通信端口号
    """
    context = zmq.Context()
    print ("Connecting to server with ports %s" % ports)
    socket = context.socket(zmq.REQ)
    for port in ports:
        socket.connect ("tcp://localhost:%s" % port)
    # 数据分发
    i = 0
    count = 0
    while i<c: 
        socket.send(str(i).encode("ascii"))
        message = socket.recv()
        # print(f"第{i}号数据处理完毕,count={message}")
        i+=1
        count += int(message)
    print(f"共上传处理{count}条数据")


if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_onnx', type=int, default=0)
    parser.add_argument('--predict_model', type=str, default='torch', choices=["torch",'onnx', 'triton'])
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
    parser.add_argument('--result_path', type= str, default= 'data')
    parser.add_argument('--max_size', type=int, default=50000, help = 'The number of generated sample data')
    args = parser.parse_args()
    t=time.time()
    config = PredictConfig(args)
    config.result_file = os.path.join(args.result_path, "result_multi_triton.txt")
    print(f"加载参数 time={get_time_dif(t)}")
    
    server_model = TritonModel()
    
    data = DataGenerator(config)
    config.vocab = data.vocab.idx2word
    inputs, tags = map(lambda x: np.array(x,dtype=int), data.load_dataset_predict(config.testfile))
    

    start=time.time()
    mlock=Lock()

    server_ports = range(5550,5560,2)
    max_size = int(args.max_size/len(server_ports))
    len_port = int(inputs.shape[0]/len(server_ports))
    f = open(config.result_file, 'w', encoding='utf-8') 
    ps = []
    server_count = int( np.ceil(max_size/ args.batch_size) )
    for i,server_port in enumerate(server_ports):
        inputs0 = inputs[i*len_port: min([i*len_port+len_port, inputs.shape[0]])]
        tags0 = tags[i*len_port: min([i*len_port+len_port, inputs.shape[0]])]
        data  = generator_data(inputs, tags, args.batch_size, max_size)
        
        pi = Process(target=server, args=(server_port, config,  server_model, data, server_count,mlock, f))
        pi.start()
        ps.append(pi)

    
    pc = Process(target=client, args=(server_ports, len(server_ports)*server_count))
    pc.start()
    pc.join()
    for pi in ps:
        pi.join()
    print(f"采用{len(ps)}个进程和预测服务通信，预计处理{args.max_size}条数据，共耗时{get_time_dif(start)}")
    f.close()