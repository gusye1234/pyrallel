
// #define GENERIC_ADD(type) type add_##type(type i, type j)
// #define GENERIC_multi(type) type multi_##type(type i, type j)
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <omp.h>
#include <math.h>

typedef int int32;
typedef long int64;
typedef float float32;
typedef double float64;
typedef long double float128;

namespace py = pybind11;
int omp_thread_count();

//  | py::array::forcecast

#define ALLTYPE(function) \
    function(int32);      \
    function(int64);      \
    function(float32);    \
    function(float64);    \
    function(float128)

#define ALLTYPE_FLOAT(function) \
    function(int32, float64);   \
    function(int64, float64);   \
    function(float32, float32); \
    function(float64, float64); \
    function(float128, float128)

// avaliable vars: ptr_in, index, scalar
#define VECTOR_SCALAR_TYPE(type, type2, OPERATOR)                                        \
    {                                                                                    \
        py::buffer_info buf = in_array.request();                                        \
        py::array_t<type2> temp = py::array_t<type2>(buf.shape);                         \
        py::buffer_info buf_temp = temp.request();                                       \
        size_t arr_size = buf.size;                                                      \
        type2 *ptr = (type2 *)buf_temp.ptr;                                              \
        type *ptr_in = (type *)buf.ptr;                                                  \
        int max_threads, arrange;                                                        \
        Arrange(arr_size, max_threads, arrange);                                         \
        _Pragma("omp parallel for") for (int thread = 0; thread < max_threads; thread++) \
        {                                                                                \
            for (int i = 0; i < arrange; i++)                                            \
            {                                                                            \
                int index = (thread * arrange + i);                                      \
                if (index >= arr_size)                                                   \
                {                                                                        \
                    break;                                                               \
                }                                                                        \
                ptr[index] = OPERATOR;                                                   \
            }                                                                            \
        }                                                                                \
        return temp;                                                                     \
    }

// avaliable vars: ptr_in, index, ptr_sec
#define VECTOR_VECTOR_TYPE(type, type2, OPERATOR)                                        \
    {                                                                                    \
        py::buffer_info buf_in = in_array.request();                                     \
        py::buffer_info buf_sec = sec_array.request();                                   \
        py::array_t<type2> temp = py::array_t<type2>(buf_in.shape);                      \
        py::buffer_info buf_temp = temp.request();                                       \
        size_t arr_size = buf_in.size;                                                   \
        type2 *ptr = (type2 *)buf_temp.ptr;                                              \
        type *ptr_in = (type *)buf_in.ptr;                                               \
        type *ptr_sec = (type *)buf_sec.ptr;                                             \
        int max_threads, arrange;                                                        \
        Arrange(arr_size, max_threads, arrange);                                         \
        _Pragma("omp parallel for") for (int thread = 0; thread < max_threads; thread++) \
        {                                                                                \
            for (int i = 0; i < arrange; i++)                                            \
            {                                                                            \
                int index = (thread * arrange + i);                                      \
                if (index >= arr_size)                                                   \
                {                                                                        \
                    break;                                                               \
                }                                                                        \
                ptr[index] = OPERATOR;                                                   \
            }                                                                            \
        }                                                                                \
        return temp;                                                                     \
    }

// avaliable vars: ptr_in, index
#define VECTOR_TYPE(type, type2, OPERATOR)                                               \
    {                                                                                    \
        py::buffer_info buf_in = in_array.request();                                     \
        py::array_t<type2> temp = py::array_t<type2>(buf_in.shape);                      \
        py::buffer_info buf_temp = temp.request();                                       \
        size_t arr_size = buf_in.size;                                                   \
        type *ptr_in = (type *)buf_in.ptr;                                               \
        type2 *ptr_temp = (type2 *)buf_temp.ptr;                                         \
        int max_threads, arrange;                                                        \
        Arrange(arr_size, max_threads, arrange);                                         \
        _Pragma("omp parallel for") for (int thread = 0; thread < max_threads; thread++) \
        {                                                                                \
            for (int i = 0; i < arrange; i++)                                            \
            {                                                                            \
                int index = (thread * arrange + i);                                      \
                if (index >= arr_size)                                                   \
                {                                                                        \
                    break;                                                               \
                }                                                                        \
                ptr_temp[index] = OPERATOR;                                              \
            }                                                                            \
        }                                                                                \
        return temp;                                                                     \
    }

#define ADD_SCALARNAME(type) py::array_t<type> add_scalar_##type(py::array_t<type, py::array::c_style> &in_array, type scalar)
#define ADD_VECTORNAME(type) py::array_t<type> add_vector_##type(py::array_t<type, py::array::c_style> &in_array, py::array_t<type, py::array::c_style> &sec_array)
#define RSUB_SCALARNAME(type) py::array_t<type> rsub_scalar_##type(py::array_t<type, py::array::c_style> &in_array, type scalar)
#define SUB_SCALARNAME(type) py::array_t<type> sub_scalar_##type(py::array_t<type, py::array::c_style> &in_array, type scalar)
#define SUB_VECTORNAME(type) py::array_t<type> sub_vector_##type(py::array_t<type, py::array::c_style> &in_array, py::array_t<type, py::array::c_style> &sec_array)
#define MUL_SCALARNAME(type) py::array_t<type> mul_scalar_##type(py::array_t<type, py::array::c_style> &in_array, type scalar)
#define MUL_VECTORNAME(type) py::array_t<type> mul_vector_##type(py::array_t<type, py::array::c_style> &in_array, py::array_t<type, py::array::c_style> &sec_array)
#define RDIV_SCALARNAME(type, type2) py::array_t<type2> rdiv_scalar_##type(py::array_t<type, py::array::c_style> &in_array, type2 scalar)
#define DIV_VECTORNAME(type, type2) py::array_t<type2> div_vector_##type(py::array_t<type, py::array::c_style> &in_array, py::array_t<type, py::array::c_style> &sec_array)
#define DIV_SCALARNAME(type, type2) py::array_t<type2> div_scalar_##type(py::array_t<type, py::array::c_style> &in_array, type2 scalar)
#define EXP_VECTORNAME(type, type2) py::array_t<type2> exp_vector_##type(py::array_t<type, py::array::c_style> &in_array)
#define SIN_VECTORNAME(type) py::array_t<float32> sin_vector_##type(py::array_t<type, py::array::c_style> &in_array)
#define COS_VECTORNAME(type) py::array_t<float32> cos_vector_##type(py::array_t<type, py::array::c_style> &in_array)
#define NEG_VECTORNAME(type) py::array_t<type> neg_vector_##type(py::array_t<type, py::array::c_style> &in_array)

py::array_t<float32> matmul(py::array_t<float32, py::array::c_style> &mat1, py::array_t<float32, py::array::f_style> &mat2);

ALLTYPE(ADD_SCALARNAME);
ALLTYPE(ADD_VECTORNAME);

ALLTYPE(RSUB_SCALARNAME);
ALLTYPE(SUB_SCALARNAME);
ALLTYPE(SUB_VECTORNAME);

ALLTYPE_FLOAT(RDIV_SCALARNAME);
ALLTYPE_FLOAT(DIV_VECTORNAME);
ALLTYPE_FLOAT(DIV_SCALARNAME);

ALLTYPE(MUL_SCALARNAME);
ALLTYPE(MUL_VECTORNAME);

ALLTYPE(NEG_VECTORNAME);

ALLTYPE_FLOAT(EXP_VECTORNAME);

ALLTYPE(SIN_VECTORNAME);

ALLTYPE(COS_VECTORNAME);

// Can you see me?
// class Matrix
// {
// public:
//     Matrix(size_t rows, size_t cols) : m_rows(rows), m_cols(cols)
//     {
//         m_data = new float[rows * cols];
//     }
//     float *data() { return m_data; }
//     size_t rows() const { return m_rows; }
//     size_t cols() const { return m_cols; }
//     float get_value(size_t row, size_t col) const {
//         return this->m_data[row*this->m_cols + col];
//     }
// private:
//     size_t m_rows, m_cols;
//     float *m_data;
// };
// py::buffer_info matrix_buff(Matrix &m);