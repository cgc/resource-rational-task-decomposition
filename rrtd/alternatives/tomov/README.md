# Setup guide

I ran the following with MATLAB 2021B.

1. Get `matlabengine` installed, here is the [help page](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html). This can bit a bit difficult to manage dependending on the environment, but generally requires `pip install $MATLAB_DIR/extern/engines/python`. Some other notes on how I got this installed at PNI are in [this document](../matlab-installation.md).

2. Check out the code for Tomov et al. 2020 into this directory

```
git clone git@github.com:tomov/chunking.git
cd chunking
git checkout 2ba7fa618a15177c6e29110aa9b7ae168339191b
cd ..
```

3. Download boost 1.64.0 from [here](https://boostorg.jfrog.io/artifactory/main/release/1.64.0/source/), unzipping into `chunking/include`.

4. Compiling the `sample_c.cpp` into a `.mex`

```
cd chunking
mex sample_c.cpp printmex.cpp -Iinclude/boost_1_64_0/
cd ..
```
