import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

from weight_regularization import orth_dist


def measure_so(w):
    start = time.time()
    _ = orth_dist(w)

    return time.time() - start


def measure_gso_inter(w, num_groups, group_size):
    start = time.time()
    dist_time = []
    for ii in range(num_groups):
        wt = w[ii * group_size: (ii + 1) * group_size]
        before = time.time()
        _ = orth_dist(wt)
        dist_time.append(time.time() - before)

    # print(f'Inter total loop time = {time.time() - start}. Dist calc time ='
    #       f' {num_groups} * {np.mean(dist_time)} = {num_groups * np.mean(dist_time)}')
    return time.time() - start


def measure_gso_intra(w, group_size):
    start = time.time()
    for ii in range(group_size):
        wt = w[ii::group_size]
        _ = orth_dist(wt)

    return time.time() - start


def main():
    # Define constants
    C_OUT = 160
    D = C_OUT * 3 * 3

    N_RUNS = 1000
    GROUP_SIZES = [2, 4, 8, 16, 32, 64]
    w = torch.randn((C_OUT, D)).cuda()

    so_runtimes = []
    gso_inter_runtimes = []
    gso_intra_runtimes = []
    for num_groups in tqdm(GROUP_SIZES):
        current_so_runtimes = []
        current_gso_inter_runtimes = []
        current_gso_intra_runtimes = []
        group_size = C_OUT // num_groups
        for _ in range(N_RUNS):
            current_so_runtimes.append(measure_so(w))
            current_gso_inter_runtimes.append(measure_gso_inter(w, num_groups, group_size))
            current_gso_intra_runtimes.append(measure_gso_intra(w, group_size))

        so_runtimes.append(np.mean(current_so_runtimes))
        gso_inter_runtimes.append(np.mean(current_gso_inter_runtimes))
        gso_intra_runtimes.append(np.mean(current_gso_intra_runtimes))

    # Plot results
    plt.plot(GROUP_SIZES, so_runtimes, label='SO')
    plt.plot(GROUP_SIZES, gso_inter_runtimes, label='GSO_inter')
    plt.plot(GROUP_SIZES, gso_intra_runtimes, label='GSO_intra')
    plt.legend()
    plt.xlabel('Group size')
    plt.ylabel('Avg runtime')
    plt.savefig('/mnt5/yoavkurtz/pytorch-classification/outputs/runtime_plot.jpg')


if __name__ == '__main__':
    main()
