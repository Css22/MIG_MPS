import argparse
import json
import numpy as np
import os
import time

from jobs import entry
file_path = ""
def store_lantency(latency_list, task, batch, SM):
    percentile_99 = np.percentile(latency_list, 99)

    data = {
        "model": task,
        "batch": batch,
        "SM": SM,
        "99th_percentile_latency": percentile_99
    }
    
    with open(file_path, 'a+') as json_file:
        json.dump(data, json_file, indent=4)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str)
    parser.add_argument("--SM", type=int)
    args = parser.parse_args()
    task = args.task
    batch = args.batch
    SM = args.SM



    if task == 'bert':  
        model = entry.get_model(task)
        model = model().half().cuda(0).eval()
    else:
        model = entry.get_model(task)
        model = model().cuda(0).eval()

    if task == 'bert':
        input,masks = entry.get_input(task, batch)
    else:
        input = entry.get_input(task, batch)


    latency_list = []


    for batch in range(0, 1000):
        for i in range(0, 100):
            start_time = time.time()
            if task == 'bert':
                output= model.run(input,masks,0,12).cpu()
            elif task == 'deeplabv3':
                output= model(input)['out'].cpu()
            else:
                output=model(input).cpu()
            end_time = time.time()
            latency_list.append((end_time - start_time) * 1000)

    store_lantency(latency_list, task, batch, SM)
      