# -*- coding: utf-8 -*-
from functools import partial
import sys
import numpy as np
import torch.nn.functional as F
import onnxruntime
from tritonclient.utils import InferenceServerException
import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient
    
class OnnxModel:
    def __init__(self, onnx_path):
        self.onet_session = onnxruntime.InferenceSession(onnx_path)

    def predict(self,data):
        """
            args:
                dataset: numpy.ndarray, [batch_size, pad_size]
            return:   
                output: numpy.ndarray, [batch_size]
        """
        inputs = {self.onet_session.get_inputs()[0].name: data}
        outputs = self.onet_session.run(["output"], inputs)
        return np.argmax(outputs[0], 1)


class TritonModel:
    def __init__(self, model_name = 'dpcnn_testtest'):
        self.triton_client = httpclient.InferenceServerClient(url='10.1.60.158:8000')
        self.model_name = model_name
    
    def predict(self, data):
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


class AsyncTritonModel:
    def __init__(self, model_name = 'dpcnn_testtest'):
        self.triton_client = httpclient.InferenceServerClient(url='10.1.60.158:8000')
        self.model_name = model_name
    
    def predict(self, data):
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

      
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def _callback(user_data, result, error):
    if error:
        user_data.append(error)
    else:
        user_data.append(result)
        