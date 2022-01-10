from . import base
from .base import TYPE_PRIORITY, base_tracer
from .utils import timer
import pycuda.gpuarray as gpu
from pycuda.elementwise import ElementwiseKernel
from pycuda.compiler import SourceModule
import numpy as np

def query_device():
    import pycuda
    import pycuda.autoinit
    import pycuda.driver as drv
    drv.init()
    print('CUDA device query (PyCUDA version) \n')
    print(f'Detected {drv.Device.count()} CUDA Capable device(s) \n')
    for i in range(drv.Device.count()):

        gpu_device = drv.Device(i)
        print(f'Device {i}: {gpu_device.name()}')
        compute_capability = float('%d.%d' % gpu_device.compute_capability())
        print(f'\t Compute Capability: {compute_capability}')
        print(
            f'\t Total Memory: {gpu_device.total_memory()//(1024**2)} megabytes'
        )

        # The following will give us all remaining device attributes as seen
        # in the original deviceQuery.
        # We set up a dictionary as such so that we can easily index
        # the values using a string descriptor.

        device_attributes_tuples = gpu_device.get_attributes().items()
        device_attributes = {}

        for k, v in device_attributes_tuples:
            device_attributes[str(k)] = v

        num_mp = device_attributes['MULTIPROCESSOR_COUNT']

        # Cores per multiprocessor is not reported by the GPU!
        # We must use a lookup table based on compute capability.
        # See the following:
        # http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities

        cuda_cores_per_mp = {
            5.0: 128,
            5.1: 128,
            5.2: 128,
            6.0: 64,
            6.1: 128,
            6.2: 128,
            7.5: 128,
        }[compute_capability]

        print(
            f'\t ({num_mp}) Multiprocessors, ({cuda_cores_per_mp}) CUDA Cores / Multiprocessor: {num_mp*cuda_cores_per_mp} CUDA Cores'
        )

        device_attributes.pop('MULTIPROCESSOR_COUNT')

        for k in device_attributes.keys():
            print(f'\t {k}: {device_attributes[k]}')

kernels = {
    "add_scalar_int32" : ElementwiseKernel("int *temp, int *a,int scalar", "temp[i] = scalar+a[i]"),
    "add_scalar_float32" : ElementwiseKernel("float *temp,float *a,float scalar", "temp[i] = scalar+a[i]"),
    "add_vector_int32" : ElementwiseKernel("int *temp, int *a, int* b", "temp[i] = b[i]+a[i]"),
    "add_vector_float32" : ElementwiseKernel("float *temp,float *a,float* b", "temp[i] = b[i]+a[i]"),
    "mul_scalar_int32" : ElementwiseKernel("int *temp, int *a, int scalar", "temp[i] = scalar*a[i]"),
    "mul_scalar_float32" : ElementwiseKernel("float *temp,float *a,float scalar", "temp[i] = scalar*a[i]"),
    "mul_vector_int32" : ElementwiseKernel("int *temp, int *a, int* b", "temp[i] = b[i]*a[i]"),
    "mul_vector_float32" : ElementwiseKernel("float *temp,float *a,float* b", "temp[i] = b[i]*a[i]"),
    "rsub_scalar_int32" : ElementwiseKernel("int *temp, int *a,int scalar", "temp[i] = scalar-a[i]"),
    "rsub_scalar_float32" : ElementwiseKernel("float *temp,float *a,float scalar", "temp[i] = scalar-a[i]"),
    "sub_scalar_int32" : ElementwiseKernel("int *temp, int *a,int scalar", "temp[i] = a[i]-scalar"),
    "sub_scalar_float32" : ElementwiseKernel("float *temp,float *a,float scalar", "temp[i] = a[i]-scalar"),
    "sub_vector_int32" : ElementwiseKernel("int *temp, int *a, int *b", "temp[i] = a[i]-b[i]"),
    "sub_vector_float32" : ElementwiseKernel("float *temp,float *a,float* b", "temp[i] = a[i]-b[i]"),
    "rdiv_scalar_int32" : ElementwiseKernel("float *temp, int *a, int scalar", "temp[i] = scalar/a[i]"),
    "rdiv_scalar_float32" : ElementwiseKernel("float *temp,float *a, float scalar", "temp[i] = scalar/a[i]"),
    "div_scalar_int32" : ElementwiseKernel("float *temp, int *a,int scalar", "temp[i] = a[i]/scalar"),
    "div_scalar_float32" : ElementwiseKernel("float *temp,float *a,float scalar", "temp[i] = a[i]/scalar"),
    "div_vector_int32" : ElementwiseKernel("float *temp, int *a, int* b", "temp[i] = a[i]/b[i]"),
    "div_vector_float32" : ElementwiseKernel("float *temp,float *a,float* b", "temp[i] = a[i]/b[i]"),

    "sin_vector_float32" : ElementwiseKernel("float *temp,float *a",
                                             "temp[i] = f(a[i])",
                                             preamble="""
                                                __device__ float f(float x){
                                                    return sin(x);
                                                }
                                             """),
    "sin_vector_int32" : ElementwiseKernel("float *temp,int *a",
                                             "temp[i] = f(a[i])",
                                             preamble="""
                                                __device__ float f(int x){
                                                    return sin(x);
                                                }
                                             """),
    "cos_vector_float32" : ElementwiseKernel("float *temp,float *a",
                                             "temp[i] = f(a[i])",
                                             preamble="""
                                                __device__ float f(float x){
                                                    return cos(x);
                                                }
                                             """),
    "cos_vector_int32" : ElementwiseKernel("float *temp,int *a",
                                             "temp[i] = f(a[i])",
                                             preamble="""
                                                __device__ float f(int x){
                                                    return cos(x);
                                                }
                                             """),
    "exp_vector_float32" : ElementwiseKernel("float *temp,float *a",
                                             "temp[i] = f(a[i])",
                                             preamble="""
                                                __device__ float f(float x){
                                                    return exp(x);
                                                }
                                             """),
    "exp_vector_int32" : ElementwiseKernel("float *temp,int *a",
                                             "temp[i] = f(a[i])",
                                             preamble="""
                                                __device__ float f(int x){
                                                    return exp(x);
                                             }
                                             """),
    "neg_vector_float32" : ElementwiseKernel("float *temp,float *a", "temp[i] = -a[i]",),
    "neg_vector_int32" : ElementwiseKernel("int *temp,int *a", "temp[i] = -a[i]"),
}


template = {
    "1_float32": ("float *temp,float *a", "temp[i] = f(a[i])", "__device__ float f(float a){{return {command};}}"),
    "2_float32": ("float *temp,float *a, float* b", "temp[i] = f(a[i], b[i])", "__device__ float f(float a, float b){{return {command};}}")
}

def check_fusion(command):
    a = b = c = 0
    def one(x):
        return x
    def two(x, y):
        return x
    exp = sin = cos = one
    eval(command)



def _OP_2(a: gpu.GPUArray, b: gpu.GPUArray, types: str, op_name, right=False, return_type=None):
    return_type = return_type or types
    with timer(name='create'):
        temp = gpu.empty(a.shape, dtype=return_type)
    if np.isscalar(b):
        if TYPE_PRIORITY[str(a.dtype)] < TYPE_PRIORITY[types]:
            a = a.astype(types)
        with timer(name='compute'):
            if right:
                kernels[f'r{op_name}_scalar_' + types](temp, a, b)
            else:
                kernels[f'{op_name}_scalar_' + types](temp, a, b)
    else:
        with timer(name='convert'):
            if TYPE_PRIORITY[str(a.dtype)] < TYPE_PRIORITY[types]:
                a = a.astype(types)
            if TYPE_PRIORITY[str(b.dtype)] < TYPE_PRIORITY[types]:
                b = b.astype(types)
        # a, b = broadcast(a, b)
        with timer(name='compute'):
            if right:
                kernel[f'{op_name}_vector_' + types](temp, b, a)
            else:
                kernels[f'{op_name}_vector_' + types](temp, a, b)
    return temp

def _OP_1(a: gpu.GPUArray, types: str, op_name, return_type=None):
    return_type = return_type or types
    with timer(name='create'):
        temp = gpu.empty(a.shape, dtype=return_type)
    with timer(name='compute'):
        kernels[f'{op_name}_vector_' + types](temp, a)
    return temp

def cuda_add(a: gpu.GPUArray, b: gpu.GPUArray, types: str):
    return _OP_2(a, b, types, 'add')

def cuda_sub(a: gpu.GPUArray, b: gpu.GPUArray, types: str, right=False):
    return _OP_2(a, b, types, 'sub', right=right)

def cuda_mul(a: gpu.GPUArray, b: gpu.GPUArray, types: str):
    return _OP_2(a, b, types, 'mul')

def cuda_div(a: gpu.GPUArray, b: gpu.GPUArray, types: str, right=False):
    return _OP_2(a, b, types, 'div', return_type='float32',right=right)

def cuda_exp(a, types):
    return _OP_1(a, types, 'exp', return_type='float32')

def cuda_cos(a, types):
    return _OP_1(a, types, 'cos', return_type='float32')

def cuda_sin(a, types):
    return _OP_1(a, types, 'sin',  return_type='float32')

def cuda_neg(a, types):
    return _OP_1(a, types, 'neg')
