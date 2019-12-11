import os

import h5py
import numpy as np
import pandas as pd

source_dir = 'results/early_stopped_fc_0.25'

seeds = [0, 1, 2, 3, 4]
final_weights_inds = [4, 1, 1, 2, 1]

index = 0
df = pd.DataFrame(
    columns=['mask_criterion', 'seed', 'iteration', 'test_accuracy'],
    index=np.arange(1625)
)

mask_criteria = [
    'large_final',
    'small_final',
    'random',
    'large_init',
    'small_init',
    'large_init_large_final',
    'small_init_small_final',
    'magnitude_increase',
    'movement',
    'large_final_diff_sign',
    'large_final_same_sign',
    'magnitude_increase_diff_sign',
    'magnitude_increase_same_sign',
]

for seed, final_weights_ind in zip(seeds, final_weights_inds):
    seed_dir = os.path.join(source_dir, 'test_seed_{}'.format(seed))

    results_file = os.path.join(seed_dir,
                                'mask_scenarios',
                                'all_init_accuracy_lot_exp_none_fw_ind_{}'.format(final_weights_ind),
                                )

    with h5py.File(results_file, 'r') as results:
        test_accuracies = results['test_accuracy'][...]
        for n, mask_criterion in enumerate(mask_criteria):
            for percentile in range(25):  # 0 as a percentile, too.
                percentile_test_acc = test_accuracies[percentile * 13 + n][0]

                df.iloc[index] = [
                    mask_criterion,
                    seed,
                    percentile,
                    percentile_test_acc,
                ]
                index += 1

df.to_csv('nips-reproducibility-challenge-results/crawled_data_early_stopped_0.25_val_acc.csv')
