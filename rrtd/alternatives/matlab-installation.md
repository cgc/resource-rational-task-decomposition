

## Installing MATLAB for Python

### On PNI Linux

#### Matlab 2019

```
# copying to a directory we have write permissions on
cp -r /jukebox/pkgs/MATLAB/R2019b5/extern/engines/python matlab2019-python-package
cd matlab2019-python-package
```

Edit `setup.py` to change two variables in `_generate_arch_file()`
```
_bin_dir='/jukebox/pkgs/MATLAB/R2019b5/bin/'
_engine_dir='/jukebox/pkgs/MATLAB/R2019b5/extern/engines/python/dist/matlab/engine/'
```

Finally run setup from within a conda env. This should correctly install it.
```
python setup.py install
```

Caveats here: on PNI's machines, it seems older MATLAB (2018 and before) hangs, and newer MATLAB (at least 2020b) uses libstdc++ headers that are too new. 2019b seems to work well, and i haven't exhaustively tried alternatives.

#### Matlab 2021 (supports Python 3.8+)

To support Python 3.8+, I worked around libstd++ header issues by setting an LD_PRELOAD to a newer version of libstdc++ installed via conda, with something like:

```
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
```

The rest of the setup is easier since the directory can be written into. So:

```
cd /jukebox/pkgs/MATLAB/R2021b/extern/engines/python
python setup.py install
```

### On Mac
I had success running `python setup.py install` in the `extern/engines/python` folder, but assigned a local prefix that I think symlinked as appropriate. Having made things work on linux I assume there is a simpler way to make this work with a virtualenv.

## Running
- chunking, using run_models.py. Requires 106 Python and 165 MATLAB threads per job (really don't know why...), so with a per-user limit of 4096 processes (`ulimit -a | grep 'max user proc'`), that means 12 or so is a good number to use. `python run_models.py tomov_analysis --n_jobs=12 --mdps experiment_mdps`
