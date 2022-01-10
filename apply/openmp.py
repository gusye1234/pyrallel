import numpy as np
from . import omp
from .base import base_tracer, broadcast, TYPE_PRIORITY


def support_omp():
    try:
        from . import omp
        return omp.openmp()
    except ModuleNotFoundError:
        return False


def omp_num_threads():
    if support_omp():
        from . import omp
        return omp.num_threads()
    else:
        return -1


def omp_set_num_threads(num):
    if support_omp():
        from . import omp
        omp.set_num_threads(num)

def _OP_2(a: np.ndarray, b: np.ndarray, types: str, op_name, right=False):
    if np.isscalar(b):
        if TYPE_PRIORITY[str(a.dtype)] < TYPE_PRIORITY[types]:
            a = a.astype(types)
        if right:
            return getattr(omp, f'r{op_name}_scalar_' + types)(a, b)
        else:
            return getattr(omp, f'{op_name}_scalar_' + types)(a, b)
    else:
        if TYPE_PRIORITY[str(a.dtype)] < TYPE_PRIORITY[types]:
            a = a.astype(types)
        if TYPE_PRIORITY[str(b.dtype)] < TYPE_PRIORITY[types]:
            b = b.astype(types)
        # a, b = broadcast(a, b)
        if right:
            return getattr(omp, f'{op_name}_vector_' + types)(b, a)
        else:
            return getattr(omp, f'{op_name}_vector_' + types)(a, b)

def _OP_1(a: np.ndarray, types: str, op_name):
    # a = np.asanyarray(a)
    return getattr(omp, f'{op_name}_vector_' + types)(a)

def omp_add(a: np.ndarray, b: np.ndarray, types: str):
    return _OP_2(a, b, types, 'add')

def omp_sub(a: np.ndarray, b: np.ndarray, types: str, right=False):
    return _OP_2(a, b, types, 'sub', right=right)

def omp_mul(a: np.ndarray, b: np.ndarray, types: str):
    return _OP_2(a, b, types, 'mul')

def omp_div(a: np.ndarray, b: np.ndarray, types: str, right=False):
    return _OP_2(a, b, types, 'div', right=right)

def omp_exp(a, types):
    return _OP_1(a, types, 'exp')

def omp_cos(a, types):
    return _OP_1(a, types, 'cos')

def omp_sin(a, types):
    return _OP_1(a, types, 'sin')

def omp_neg(a, types):
    return _OP_1(a, types, 'neg')