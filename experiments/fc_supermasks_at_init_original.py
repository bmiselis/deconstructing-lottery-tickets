import os
from multiprocessing.pool import Pool


seeds = [0, 1, 2, 3, 4]


def run_seed(seed):
    os.system('python get_init_loss_train_lottery.py --output_dir ./results/iter_lot_fc_orig/test_seed_{}/ --train_h5 ./data/mnist_train.h5 --test_h5 ./data/mnist_test.h5 --arch fc_lot --seed {} --opt adam --lr 0.0012 --exp none --layer_cutoff 4,6 --prune_base 0.8,0.9 --prune_power 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24'.format(seed, seed))


# Run all trainings in parallel.
with Pool(processes=5) as pool:
    pool.map(run_seed, seeds)

