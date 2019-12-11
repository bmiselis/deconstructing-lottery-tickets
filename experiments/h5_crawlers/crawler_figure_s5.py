import os

import h5py
import numpy as np
import pandas as pd

source_dir = 'results/iter_lot_fc_orig'

seeds = [0, 1, 2, 3, 4]

index = 0
df = pd.DataFrame(
    columns=['mask_criterion', 'seed', 'iteration', 'test_accuracy'],
    index=np.arange(375)
)

criteria_to_extract = [
    'large_final',
    'large_final_diff_sign',
    'large_final_same_sign',
]

for seed in seeds:
    seed_dir = os.path.join(source_dir, 'test_seed_{}'.format(seed))
    mask_criteria = [el for el in os.listdir(seed_dir)
                     if os.path.isdir(os.path.join(seed_dir, el))]

    original_weights_file = os.path.join(seed_dir, 'weights')

    with h5py.File(original_weights_file, 'r') as data:
        idx_min_val_loss = np.argmin(data['val_loss'][...])
        original_test_accuracy = data['test_accuracy'][...][idx_min_val_loss]

    for mask_criterion in mask_criteria:
        if mask_criterion[:-27] not in criteria_to_extract:
            continue

        mask_dir = os.path.join(seed_dir, mask_criterion)
        iterations = os.listdir(mask_dir)
        iterations = list(map(lambda x: x[2:], iterations))
        iterations.insert(0, 0)  # Placeholder for unpruned network accuracy.
        iterations = sorted(iterations, key=int)

        for n, iteration in enumerate(iterations, start=0):
            if n == 0:
                df.iloc[index] = [
                    mask_criterion[:-27],
                    seed,
                    0,
                    original_test_accuracy[0],
                ]
                index += 1
            else:
                iteration_dir = os.path.join(mask_dir, 'pp' + iteration)
                weights_file = os.path.join(iteration_dir, 'weights')
                with h5py.File(weights_file, 'r') as data:
                    idx_min_val_loss = np.argmin(data['val_loss'][...])
                    test_accuracy = data['test_accuracy'][...][idx_min_val_loss]
                    df.iloc[index] = [
                        mask_criterion[:-27], seed, n, test_accuracy[0],
                    ]
                    index += 1

df.to_csv('results/crawled_data_figure_s5.csv')
