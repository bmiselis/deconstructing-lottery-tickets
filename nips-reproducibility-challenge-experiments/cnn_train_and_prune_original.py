import itertools
import os
from multiprocessing.pool import Pool

seeds = [0, 1, 2, 3, 4]
networks = ['conv2', 'conv4', 'conv6']

mask_criteria = [
    'magnitude_increase',
    'movement',
    'large_final',
    'small_final',
    'large_init',
    'small_init',
    'large_init_large_final',
    'small_init_small_final',
    'random',
    'large_final_diff_sign',
    'large_final_same_sign',
    'magnitude_increase_diff_sign',
    'magnitude_increase_same_sign',
]


def train_unpruned(seed, network):
    command = "bash print_train_command.sh iter {} test {} t".format(network,
                                                                     seed)
    os.system(command)


def iterative_pruning(seed, network, mask_criterion):
    command = "bash print_train_lottery_iterative_command.sh {} test {} {} -1 mask none t".format(network, seed, mask_criterion)
    os.system(command)


# CNNs on cifar10 need to be trained one by one, otherwise there would be
# memory issues.
with Pool(processes=1) as pool:
    all_params = list(itertools.product(seeds, networks))
    pool.starmap(train_unpruned, all_params)

# Run iterative pruning (also one by one because of memory issues).
with Pool(processes=1) as pool:
    all_params = list(itertools.product(seeds, networks, mask_criteria))
    pool.starmap(iterative_pruning, all_params)
