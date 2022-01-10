## Apply

A developing library for accelerating part of  `pandas` using parallelizing program.

Target to support:

1. Operator with c++ extension, (`Openmp`, `pybind11`)
2. Operator with cuda extension, (`pycuda`, `numba`)



### To set up

If you're in linux platform, make sure you install `openmp`, run

```
apt install libomp-dev
```

If you're in macOS, run

```
brew install llvm libomp
```

Run

```
pip install -r requirements.txt
```



Under `Apply/`, 

* run `make build` to build the extension. 
* To install, run `make install` then.

