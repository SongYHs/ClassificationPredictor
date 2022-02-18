from functools import partial
from tritonclient.utils import InferenceServerException
import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient
import numpy as np
import time
import sys

def _callback(user_data, result, error):
    if error:
        user_data.append(error)
    else:
        user_data.append(result)

       
def _callback2(lock, user_data, result, error):
    print(f"开始预测{len(user_data)}*******************************************************")
    lock.acquire()
    if error:
        user_data.append(error)
    else:
        user_data.append(result)
        
    lock.release()


class TritonModel:
    def __init__(self, model_name = 'dpcnn_testtest', enterpoint='10.1.60.158:8000', is_grpc=True):
        if is_grpc:
            self.triton_client = grpcclient.InferenceServerClient(url=enterpoint)
            self.predict = self.predict_grpc
        else:
            self.triton_client = httpclient.InferenceServerClient(url=enterpoint)
            self.predict = self.predict_http
        self.model_name = model_name
    
    def predict_http(self, data):
        """
            args:
                dataset: numpy.ndarray, [batch_size, pad_size]
            return:   
                output: numpy.ndarray, [batch_size]
        """
        inputs=[]
        inputs.append(httpclient.InferInput('input', data.shape, "INT64"))
        # inputs[0].set_data_from_numpy(data)
        inputs[0].set_data_from_numpy(data, binary_data=False)
        outputs = []
        outputs.append(httpclient.InferRequestedOutput('output',binary_data=False))  # 获取 1000 维的向量

        results = self.triton_client.async_infer(self.model_name, inputs=inputs, outputs=outputs)

        return np.argmax(results.get_result().as_numpy('output'), 1)
    
    def predict_grpc(self, data):
        """
            args:
                dataset: numpy.ndarray, [batch_size, pad_size]
            return:   
                output: numpy.ndarray, [batch_size]
        """
        inputs=[]
        inputs.append(grpcclient.InferInput('input', data.shape, "INT64"))
        # inputs[0].set_data_from_numpy(data)
        inputs[0].set_data_from_numpy(data)
        outputs = []
        outputs.append(grpcclient.InferRequestedOutput('output'))  # 获取 1000 维的向量

        user_data =[]
        self.triton_client.async_infer(self.model_name, inputs=inputs, outputs=outputs, callback=partial(_callback, user_data))
        time_out = 40
        while ((len(user_data) == 0) and time_out > 0):
            time_out = time_out - 1
            time.sleep(0.5)
        if ((len(user_data) == 1)):
        # Check for the errors
            if type(user_data[0]) == InferenceServerException:
                print(user_data[0])
                sys.exit(1)
            
        return np.argmax(user_data[0].as_numpy('output'), 1)
    
    async def predict_grpc_async(self, data, user_data, request_id, lock):
        """
            args:
                dataset: numpy.ndarray, [batch_size, pad_size]
            return:   
                output: numpy.ndarray, [batch_size]
        """
        inputs=[]
        inputs.append(grpcclient.InferInput('input', data.shape, "INT64"))
        # inputs[0].set_data_from_numpy(data)
        inputs[0].set_data_from_numpy(data)
        outputs = []
        outputs.append(grpcclient.InferRequestedOutput('output'))  # 获取 1000 维的向量

        self.triton_client.async_infer(self.model_name, inputs=inputs, outputs=outputs, callback=partial(_callback2, lock, user_data), request_id=request_id)