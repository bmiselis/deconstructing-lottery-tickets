# NeurIPS 2019 Reproducibility Challenge Experiments

As part of the Applied Machine Learning course (COMP551) at McGill University,
Canada, our team conducted a set of experiments to verify whether the results
presented in the paper are reproducible.

## Authors

Numa Karolinski, Bartosz Miselis, Monisha Shcherbakova

Contact: {numa.karolinski,bartosz.miselis,monisha.shcherbakova}@mail.mcgill.ca

## Steps to Reproduce the Results

To ensure the code was runnable in our environment (Ubuntu 18.04 with 2 x NVIDIA
GTX 1080 Ti onboard) we had to perform the following steps:

1. Install `GitResultsManager`: https://github.com/yosinski/GitResultsManager.
2. Restrict `tf.InteractiveSession` from preallocating all available GPU memory
to run the experiments in parallel on a single GPU.

Next steps included:
1. Add multiple crawler scripts (can be found in
`nips-reproducibility-challenge/h5_crawlers` directory) to generate CSV files
(`nips-reproducibility-challenge-results` directory)
with the data needed to plot the figures for our report
(`nips-reproducibility-challenge-results/reproducibility_report_figures.ipynb`
notebook).
2. Run the set of experiments described in our report.

## How to Run Our Code

Before running the scripts ensure all the paths are correct for your
environment (they may need adaptation to your environment). Besides paths,
however, everything should be runnable as is.

The scripts to run main experiments are located in
`nips-reproducibility-challenge-experiments` directory and can be simply run
by executing them with `python` command:

```bash
python nips-reproducibility-challenge-experiments/fc_train_and_prune_original.py
```

Note: All the scripts for experiments and crawlers should be run from main
repository directory.