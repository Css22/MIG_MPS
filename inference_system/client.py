import tritonclient.http as httpclient
import numpy as np
import time
import threading
import math
import argparse
import random
from PIL import Image
import os

QoS_map = {
    'resnet50': 108
}
max_RPS_map = {
    'resnet50': 700
}
min_RPS_map = {
    'resnet50': 300
}

input_list = []
TRITON_SERVER_URL = "localhost:8000"
MODEL_NAME = ""
image_dir = "/data/zbw/inference_system/MIG_MPS/inference_system/dataset/test"
triton_client = httpclient.InferenceServerClient(url=TRITON_SERVER_URL)

def select_random_batch(images_array, n):
    selected_images = np.random.choice(images_array.shape[0], n, replace=False)
    return images_array[selected_images]

def load_random_batch(image_dir, batch_size):
    images = []
    filenames = random.sample(os.listdir(image_dir), batch_size)
    for filename in filenames:
        img_path = os.path.join(image_dir, filename)
        img = Image.open(img_path).resize((224, 224)) 
        img = img.convert('RGB') 
        img = np.array(img).astype(np.float32) / 255.0  
        img = np.transpose(img, (2, 0, 1))  
        images.append(img)
    return np.array(images)
    
def generate_input_data(QoS, RPS):
    half_QoS = QoS/2
    batch = math.floor(RPS/1000 * half_QoS)
    input_data = load_random_batch(image_dir, batch)

    return input_data



def get_p95(data):
    data = np.array(data)[200:]
    percentile_95 = np.percentile(data, 95)
    return percentile_95

def record_result(path, config, result):
    with open(path, 'a+') as file:
        file.write(f"Config: {config}, RPS: {result}\n")
        file.close()

def send_request(model_name, SM):
    MODEL_NAME = model_name
    min_RPS = min_RPS_map.get(model_name)
    max_RPS = max_RPS_map.get(model_name)
    QoS = QoS_map.get(model_name)
  

    for i in range(min_RPS, max_RPS+1, 10):
        RPS = i
        half_QoS = QoS/2
        tail_latecy = []
        pre_time = time.time()
    
        input_data = load_random_batch(QoS, RPS)

        inputs = []
        outputs = []

        inputs.append(httpclient.InferInput("x.1", input_data.shape, "FP32"))
        outputs.append(httpclient.InferRequestedOutput("50"))
        inputs[0].set_data_from_numpy(input_data)
        end_time = time.time()
        print(f'pre: {(end_time - pre_time) * 1000}')
        for j in range(0, 100):

            start_time = time.time()

            results = triton_client.infer(model_name=MODEL_NAME, inputs=inputs, outputs=outputs)

            end_time = time.time()

            response_time = end_time - start_time
            print(response_time * 1000)
            tail_latecy.append(response_time * 1000)
        print("next!")
        # p95 = get_p95(tail_latecy)
           
        # print(p95, i, half_QoS)
        # if p95 > half_QoS:
        #     file_path = f'/data/zbw/inference_system/MIG_MPS/log/{model_name}_MPS_QPS'
        #     record_result(path=file_path, config=SM, result=RPS-10)
        #     break
        # else:
        #     continue
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str)
    parser.add_argument("--config", default='', type=str)
    args = parser.parse_args()

    task = args.task
    config = args.config
    QoS = QoS_map.get(task)
    RPS = max_RPS_map.get(task)
    half_QoS = QoS/2
    batch = math.floor(RPS/1000 * half_QoS)
    input_list = load_random_batch(image_dir=image_dir, batch_size=batch)
    send_request(model_name=task, SM=config)