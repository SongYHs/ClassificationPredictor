import os



def data_genener(datadir, files, batch_size):
    """
        读数据迭代器
        args:
            files: 所有需要读取的数据名
            batch_size: 数据块大小
        return:
            data_id: 数据块号
            data_count: 数据块大小
            data: 数据
            output_name: 数据处理后的写入地址
            file_count: 对应写入地址应写入的文件块数目
    """
    data_id = 0
    for i, filename in enumerate(files):
        f=open(os.path.join(datadir, filename), 'r', encoding='utf-8')   
        data =''
        data_count, fc = 0, 0
        for start, line in enumerate(f.readlines()):
            data += line
            data_count += 1
            if not (start+1)%batch_size:
                # data = data.encode()
                yield str(data_id), data_count, data, filename, -1
                data = ''
                data_id += 1
                data_count = 0
                fc += 1
        if (start+1)% batch_size: 
            fc+=1     
            # data = data.encode()
            yield str(data_id), data_count, data, filename, fc
            data_id += 1
            data = ''
        f.close()