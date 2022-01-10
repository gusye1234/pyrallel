#include "operator.h"

using namespace std;
namespace py = pybind11;

// py::buffer_info matrix_buff(Matrix &m)
// {
//     return py::buffer_info(
//         m.data(),                               /* Pointer to buffer */
//         sizeof(float),                          /* Size of one scalar */
//         py::format_descriptor<float>::format(), /* Python struct-style format descriptor */
//         2,                                      /* Number of dimensions */
//         {m.rows(), m.cols()},                   /* Buffer dimensions */
//         {sizeof(float) * m.cols(),              /* Strides (in bytes) for each index */
//          sizeof(float)});
// }
void Arrange(size_t arr_size, int &max_threads, int &arrange){
    max_threads = omp_thread_count();
    arrange = (int)std::ceil((float)arr_size / max_threads);
}

#define ADD_SCALAR(type) \
    ADD_SCALARNAME(type) \
    VECTOR_SCALAR_TYPE(type, type, ptr_in[index] + scalar)

#define ADD_VECTOR(type) \
    ADD_VECTORNAME(type) \
    VECTOR_VECTOR_TYPE(type, type, ptr_in[index] + ptr_sec[index])

#define SUB_SCALAR(type) \
    SUB_SCALARNAME(type) \
    VECTOR_SCALAR_TYPE(type, type, ptr_in[index] - scalar)

#define RSUB_SCALAR(type) \
    RSUB_SCALARNAME(type) \
    VECTOR_SCALAR_TYPE(type, type, scalar - ptr_in[index])

#define SUB_VECTOR(type) \
    SUB_VECTORNAME(type) \
    VECTOR_VECTOR_TYPE(type, type, ptr_in[index] - ptr_sec[index])

#define MUL_SCALAR(type) \
    MUL_SCALARNAME(type) \
    VECTOR_SCALAR_TYPE(type, type, ptr_in[index] * scalar)

#define MUL_VECTOR(type) \
    MUL_VECTORNAME(type) \
    VECTOR_VECTOR_TYPE(type, type, ptr_in[index] * ptr_sec[index])

#define RDIV_SCALAR(type, type2) \
    RDIV_SCALARNAME(type, type2) \
    VECTOR_SCALAR_TYPE(type, type2, scalar / ptr_in[index])

#define DIV_SCALAR(type, type2) \
    DIV_SCALARNAME(type, type2) \
    VECTOR_SCALAR_TYPE(type, type2, ptr_in[index] / scalar)

#define DIV_VECTOR(type, type2) \
    DIV_VECTORNAME(type, type2) \
    VECTOR_VECTOR_TYPE(type, type2, ptr_in[index] / ptr_sec[index])

#define NEG_VECTOR(type) \
    NEG_VECTORNAME(type) \
    VECTOR_TYPE(type, type, -ptr_in[index])

#define EXP_VECTOR(type, type2) \
    EXP_VECTORNAME(type, type2) \
    VECTOR_TYPE(type, type2, std::exp(ptr_in[index]))

#define COS_VECTOR(type) \
    COS_VECTORNAME(type) \
    VECTOR_TYPE(type, float32, std::cos(ptr_in[index]))

#define SIN_VECTOR(type) \
    SIN_VECTORNAME(type) \
    VECTOR_TYPE(type, float32, std::sin(ptr_in[index]))

py::array_t<float32> matmul(py::array_t<float32, py::array::c_style> &mat1, py::array_t<float32, py::array::f_style> &mat2)
{
    py::buffer_info buf_1 = mat1.request(), buf_2 = mat2.request();
    vector<ssize_t> size_1 = buf_1.shape, size_2 = buf_2.shape;
    ssize_t row = size_1[0], col = size_2[1], inter = size_1[1];
    py::array_t<float32> temp({row, col});
    py::buffer_info buf = temp.request();
    float32 * ptr = (float32 *)buf.ptr, *ptr_1 = (float32 *)buf_1.ptr, *ptr_2 = (float32 *)buf_2.ptr;
    printf("SIZE %ld %ld %ld", row, inter, col);
    ssize_t i=0, j=0, k=0;
    #pragma omp parallel for shared(ptr, ptr_1, ptr_2, row, col, inter) private(i,j,k)
    for(i=0; i<row; i++){
        for (j = 0; j < col; j++)
        {
            ssize_t temp_i = (i*col + j);
            ptr[temp_i] = 0;
            for (k = 0; k < inter; k++)
            {
                // ptr[temp_i] = ptr[temp_i]  + ptr_1[i * inter + k] * ptr_2[k * col + j];
                ptr[temp_i] = ptr[temp_i] + ptr_1[i * inter + k] * ptr_2[j * inter + k];
            }
        }
    }
    return temp;
}

ALLTYPE(ADD_SCALAR);
ALLTYPE(ADD_VECTOR);

ALLTYPE(MUL_SCALAR);
ALLTYPE(MUL_VECTOR);

ALLTYPE(SUB_SCALAR);
ALLTYPE(SUB_VECTOR);
ALLTYPE(RSUB_SCALAR);

ALLTYPE_FLOAT(DIV_SCALAR);
ALLTYPE_FLOAT(RDIV_SCALAR);
ALLTYPE_FLOAT(DIV_VECTOR);
// ALLTYPE(RDIV_SCALAR);
ALLTYPE_FLOAT(EXP_VECTOR);

ALLTYPE(NEG_VECTOR);
ALLTYPE(COS_VECTOR);
ALLTYPE(SIN_VECTOR);
