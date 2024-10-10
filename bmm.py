import torch
import time
import pandas as pd

batch_sizes = [1, 2, 4, 8, 16, 32]
matrix_sizes = [1000, 2000, 4000]

import psutil

current_process = psutil.Process()

def track_memory_usage(interval=1, timeout=10, memory_usage=[]):
    
    for _ in range(2):
        mem_info = current_process.memory_info()
        
        memory_usage.append(mem_info.rss / (1<<30))
        time.sleep(interval)
    
result_data = []

for batch_size in batch_sizes:
    for matrix_size in matrix_sizes:
        # batch1 = torch.randn(batch_size, matrix_size, matrix_size, device="mps")
        # batch2 = torch.randn(batch_size, matrix_size, matrix_size, device="mps")
        # batch1 = torch.randn(batch_size, matrix_size, matrix_size, device="mps", dtype=torch.float16)
        # batch2 = torch.randn(batch_size, matrix_size, matrix_size, device="mps", dtype=torch.float16)
        batch1 = torch.randn(batch_size, matrix_size, matrix_size, device="mps", dtype=torch.bfloat16)
        batch2 = torch.randn(batch_size, matrix_size, matrix_size, device="mps", dtype=torch.bfloat16)
        import threading
        # breakpoint()
        
        timeout = 10  
        interval = 0.03   
        memory_usage = []
        track_thread = threading.Thread(target=track_memory_usage, args=(interval, timeout, memory_usage))
        
        track_thread.start()
        start_time = time.time()

        result = torch.bmm(batch1, batch2)  

        track_thread.join()

        mem_footprint = max(memory_usage)
        result_data.append([batch_size, matrix_size, time.time()-start_time, mem_footprint])

df = pd.DataFrame(result_data, columns=['batch_size', 'tensor_shape', 'time', 'memory'])
df.to_csv("benchmark.csv", index=False)
 