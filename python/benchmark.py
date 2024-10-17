import math
import torch
import time
import numpy as np


N_warmup = 8
N_iter_bench = 80
N_iter_func = 5

def bench(f, a, b):
    for i in range(N_warmup):
        f(a, b)
    torch.cuda.synchronize()

    s = time.perf_counter_ns()
    for i in range(N_iter_bench):
        f(a, b)
    e = time.perf_counter_ns()
    return (e - s) * 1e-6


@torch.no_grad()
def gemm_nn_torch(a, b):
    ys = []
    for i in range(N_iter_func):
        y = a @ b
        ys.append(y)
    torch.cuda.synchronize()
    return ys

def bench_shape(B, M, N, K, np_dtype, transpose="nn"):
    shape_a = (B, M, K) if transpose[0] == "n" else (B, K, M)
    shape_b = (B, K, N) if transpose[1] == "n" else (B, N, K)

    a_np = np.random.normal(0.0, 1.0 / math.sqrt(M + K), shape_a).astype(np_dtype)
    b_np = np.random.normal(0.0, 1.0 / math.sqrt(N + K), shape_b).astype(np_dtype)
   

    a_pt = torch.from_numpy(a_np).to("cuda")
    b_pt = torch.from_numpy(b_np).to("cuda")

    torch.cuda.synchronize()

    
    time_torch = bench(gemm_nn_torch, a_pt, b_pt)


    return time_torch





if __name__  == "__main__":
    dtypes = ("float32", "float16")
    #transposes = ("nn", "nt", "tn")
    

    for dtype in dtypes:
        for B in [2, 4, 8, 16, 32]:
            for S in [3000, 4000, 5000, 6000, 7000]:
                np_dtype = getattr(np, dtype)
                time_torch = bench_shape(B, S, S, S, np_dtype)

                #gflop_count = get_gflop_count(B, M, N, K)
                #gflops_mx = gflop_count / (time_mlx)
                #gflops_pt = gflop_count / (time_torch)
                #diff = gflops_mx / gflops_pt - 1.0

                print(
                    f"{B:3d}, {S:4d}, {dtype}, {time_torch/(N_iter_bench*N_iter_func)}"
                )
                #if gflops_pt >= 2.0 * gflops_mx:
                    #print("ATTENTION ^^^^^^^")

