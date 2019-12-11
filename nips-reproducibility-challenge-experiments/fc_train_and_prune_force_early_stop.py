import itertools
import os
from multiprocessing.pool import Pool

# First experiment to measure how long it takes to evaluate a single seed.
seeds = [0, 1, 2, 3, 4]

early_stop_val_accuracies = [0.25, 0.5, 0.75]

# Force early stop iterations manually extracted from `diary` files of
# corresponding experiments.
final_weights_inds = {
    0.25: [4, 1, 1, 2, 1],
    0.5: [6, 5, 3, 5, 5],
    0.75: [13, 15, 13, 15, 11]
}

def train_unpruned_force_early_stop(seed, early_stop_val_acc):
    command = "bash print_train_command_force_early_stop.sh iter fc test {} t {}".format(seed, early_stop_val_acc)
    os.system(command)


def iterative_pruning(seed, early_stop_val_acc):
    final_weights_ind = final_weights_inds[early_stop_val_acc][seed]
    command = "python get_init_loss_train_lottery.py --output_dir ./results/early_stopped_fc_{}/test_seed_{}/ --train_h5 ./data/mnist_train.h5 --test_h5 ./data/mnist_test.h5 --arch fc_lot --seed 0 --opt adam --lr 0.0012 --exp none --layer_cutoff 4,6 --prune_base 0.8,0.9 --prune_power 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24 --final_weights_ind {}".format(early_stop_val_acc, seed, final_weights_ind)
    os.system(command)


# Run all trainings in parallel.
with Pool(processes=5) as pool:
    all_params = list(itertools.product(seeds, early_stop_val_accuracies))
    pool.starmap(train_unpruned_force_early_stop, all_params)


# Run all FC mnist experiments in parallel (for efficiency reasons), our PC
# has 12 threads available.
with Pool(processes=12) as pool:
    all_params = list(itertools.product(seeds, early_stop_val_accuracies))
    pool.starmap(iterative_pruning, all_params)

