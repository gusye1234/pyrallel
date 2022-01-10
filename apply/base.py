import numpy as np
import os
try:
    import pycuda
    import pycuda.autoinit
    import pycuda.gpuarray as gpu
    CUDA_SUPPORT = True
except ModuleNotFoundError:
    CUDA_SUPPORT = False

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

SUPPORT_TYPE = ['int32', 'int64', 'int','float32', 'float64', 'float','float128']

TYPE_PRIORITY = {SUPPORT_TYPE[i]: i for i in range(len(SUPPORT_TYPE))}


class base_tracer:
    def __init__(self,
                 data : np.ndarray,
                 device_name='omp',
                 need_to_move=False):
        if np.isscalar(data):
            self._data = data
            self._scalar = True
            if isinstance(data, int):
                self.dtype = 'int32'
            elif isinstance(data, float):
                self.dtype = 'float32'
        else:
            try:
                assert str(data.dtype) in SUPPORT_TYPE
            except:
                raise NotImplementedError(f"Not support this number type {data.dtype} among {SUPPORT_TYPE}")
            self.dtype = str(data.dtype)
            if self.dtype == 'int':
                self.dtype = 'int64'
            if self.dtype == 'float':
                self.dtype = 'float64'
            self._data = data
            self._scalar = False
        self._device = device(device_name)
        self.move_data_to_()

    def get_device(self):
        return self._device

    def isscalar(self):
        return self._scalar

    def numpy(self):
        return self._data

    def move_data_to_(self):
        if (self._device == 'cuda') and (not isinstance(self._data, gpu.GPUArray)) and (not self._scalar):
            self._data = gpu.to_gpu(self._data)
            assert str(self._data.dtype) in ['int32', 'float32']
        elif (self._device == 'omp') and (not isinstance(self._data, np.ndarray)) and (not self._scalar):
            self._data = self._data.get()

    def __repr__(self):
        if self._scalar:
            array_name = f"tracer({self._data}"
        else:
            array_name = 'tracer' + repr(self._data)[5:-1]
        info = f", {self._device})"
        return f"{array_name}{info}"

    def __setitem__(self, b, c):
        self._data[b] = c

    def __array__(self):
        return self._data

class device:
    def __init__(self, name):
        assert str(name) in ['numpy', 'omp', 'cuda']
        if str(name) == 'omp' and not support_omp():
            raise TypeError(f"Not support openmp")
        elif str(name) == 'cuda' and not CUDA_SUPPORT:
            print(f"Not support Cuda, please install CUDA and pyCUDA, back to openmp")
            name = 'omp'
        self._name = str(name)

    def __repr__(self):
        return f"device({self._name})"

    def __str__(self):
        return self._name

    def __eq__(self, b):
        if isinstance(b, str):
            return self._name == b
        elif isinstance(b, device):
            return self._name == b._name
        else:
            return id(self) == id(b)

def broadcast(a, b):
    a, b = np.broadcast_arrays(a, b)
    return np.ascontiguousarray(a), np.ascontiguousarray(b)

def match_types(a : base_tracer, b : base_tracer):
    if TYPE_PRIORITY[a.dtype] > TYPE_PRIORITY[b.dtype]:
        return a.dtype
    else:
        return b.dtype

def support_omp():
    try:
        from . import omp
        return omp.openmp()
    except ModuleNotFoundError:
        return False
