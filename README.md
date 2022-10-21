# Code for *Humans decompose tasks by trading off utility and computational cost*

# Setup guide

1. Using Python 3.8 or 3.9, install packages in `requirements.txt`.
2. Using R 4.1, install packages in `DESCRIPTION`.
3. Follow steps to set up Solway et al. (2014) model [here](rrtd/alternatives/solway/README.md).
4. Follow steps to set up Tomov et al. (2020) model [here](rrtd/alternatives/tomov/README.md).

# Code structure

This is a quick guide to the structure of the code. I note locations of data, and the entrypoints that compute the figures in the paper.

- Participant data is in `rrtd/experiment/analyze/.data/1.14`. Contains raw trial data (`trialdata.csv`), experimental condition (`questiondata.csv`), and experimental configuration for psuedo-random trial assignment (`configuration.json`).
- `rrtd/journal/openfield.ipynb` - Figure 2
- `rrtd/journal/graph-heatmap.ipynb` - Figures 3, 7, 10, A1
- `rrtd/journal/model-corr.ipynb` - Figure 4
- `rrtd/journal/behavior.ipynb` - Misc. Results, Figure 6
- `rrtd/journal/predict-behavior-with-model.ipynb` - Tables 2, A1, A2, A3 and Figures 8, 9, A2, A3, A4
- `rrtd/journal/rw-theory.ipynb` - Figure A5
- Experiment code is available here: https://github.com/cgc/cocosci-optdisco.

## Running models

While the above code will transparently run models, some analyses can take quite some time -- in particular, running the Solway analysis over all graphs took somewhere between 2000-4000 hours of CPU-time. Models can be run in parallel using the command-line interface to `run_models.py`, and the model results will be cached in directories starting with `cache_run_models-v2`. Some examples of the CLI:

To run the Tomov model with 50 CPUs (Should take some care -- running MATLAB with high parallelism via Python can cause crashes):
```
python run_models.py explicit_compute --n_jobs=50 --model_filter=tomov
```

To run the Solway model with 100 CPUs:
```
python run_models.py --n_jobs=100 --model_filter=solway
```

To run all other models:
```
python run_models.py --n_jobs=100 --model_filter=~solway
```

# Running tests

```
make test
make integration-test
```
