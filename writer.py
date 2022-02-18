import os


def write_fun(data_dir, out_name, raw, result):
    """
        将结果写入dta_dir/out_name
        args:
            data_dir: 数据保存目录
            out_name: 追加写入文件名
            raw:      二进制原始数据流
            result:   二进制结果数据流
    """
    with open(os.path.join(data_dir,out_name), 'w+') as f:
        for x,y in zip(raw.split("\n"), result):
            f.write(x+"\t"+y+"\n")