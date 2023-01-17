import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

from weight_regularization import orth_dist, inter_opt, intra_opt, group_reg_ortho_l2, group_reg_ortho_l2_naive, get_layers_to_regularize

OPT = True

def measure_so(w):
    start = time.time()
    _ = orth_dist(w)

    return time.time() - start


def measure_gso_inter(w, group_size, num_groups):
    start = time.time()
    dist_time = []

    if OPT:
        before = time.time()
        _ = inter_opt(w.reshape(w.shape[0], -1), group_size, num_groups)
        dist_time.append(time.time() - before)
    else:
        for ii in range(num_groups):
            wt = w[ii * group_size: (ii + 1) * group_size]
            before = time.time()
            _ = orth_dist(wt)
            dist_time.append(time.time() - before)

    # print(f'Inter total loop time = {time.time() - start}. Dist calc time ='
    #       f' {num_groups} * {np.mean(dist_time)} = {num_groups * np.mean(dist_time)}')
    return time.time() - start


def measure_gso_intra(w, group_size, num_groups):
    start = time.time()
    if OPT:
        _ = intra_opt(w.reshape(w.shape[0], -1), group_size, num_groups)
    else:
        for ii in range(group_size):
            wt = w[ii::group_size]
            _ = orth_dist(wt)

    return time.time() - start


def main():
    # Define constants
    C_OUT = 256
    D = C_OUT * 3 * 3

    N_RUNS = 1000
    GROUP_SIZES = [2, 4, 8, 16, 32, 64]
    w = torch.randn((C_OUT, D)).cuda()

    so_runtimes = []
    gso_inter_runtimes = []
    gso_intra_runtimes = []
    for group_size in tqdm(GROUP_SIZES):
        current_so_runtimes = []
        current_gso_inter_runtimes = []
        current_gso_intra_runtimes = []
        num_groups = C_OUT // group_size
        for _ in range(N_RUNS):
            current_so_runtimes.append(measure_so(w))
            current_gso_inter_runtimes.append(measure_gso_inter(w, group_size, num_groups))
            current_gso_intra_runtimes.append(measure_gso_intra(w, group_size, num_groups))

        so_runtimes.append(np.mean(current_so_runtimes))
        gso_inter_runtimes.append(np.mean(current_gso_inter_runtimes))
        gso_intra_runtimes.append(np.mean(current_gso_intra_runtimes))

    # Plot results
    plt.plot(GROUP_SIZES, so_runtimes, label='SO')
    plt.plot(GROUP_SIZES, gso_inter_runtimes, label='GSO_inter')
    plt.plot(GROUP_SIZES, gso_intra_runtimes, label='GSO_intra')
    plt.legend()
    plt.title(f'runtime vs group size. C_out = {C_OUT}')
    plt.xlabel('Group size')
    plt.ylabel('Avg runtime')
    plt.savefig('/mnt5/yoavkurtz/GroupOrtho/pytorch-classification/outputs/runtime_plot.jpg')


def check_same_value():
    # Define constants
    C_OUT = 256
    D = C_OUT * 3 * 3
    group_size = 32

    # torch.manual_seed(0)

    model = torch.nn.Conv2d(C_OUT, C_OUT, 3).cuda()
    layers_dict = get_layers_to_regularize(model, lambda x: group_size)

    old_intra = group_reg_ortho_l2_naive(model, reg_type='intra', layers_dict=layers_dict)
    new_intra = group_reg_ortho_l2(model, reg_type='intra', layers_dict=layers_dict)
    if old_intra != new_intra:
        print(f'intra old : {old_intra} new {new_intra} diff = {abs(old_intra - new_intra)}')
    else:
        print('intra pass!')
    old_inter = group_reg_ortho_l2_naive(model, reg_type='inter', layers_dict=layers_dict)
    new_inter = group_reg_ortho_l2(model, reg_type='inter', layers_dict=layers_dict)
    if old_inter != new_inter:
        print(f'inter old : {old_inter} new {new_inter} diff = {abs(old_inter - new_inter)}')
    else:
        print(f'inter pass!')




if __name__ == '__main__':
    check_same_value()
    main()
