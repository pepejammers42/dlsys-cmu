#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;

void matmul(const float *A, const float *B, float *res, size_t M, size_t N, size_t K){
    /*
     * a helper function for matrix multiplication, res = A * B. 
     * A = M * N, B = N * K for result matrix of dim M * K.
     */
    for (size_t m = 0; m < M; m++){
        for (size_t k = 0; k < K; k++){
            res[m * K + k] = 0;
            for (size_t n = 0; n < N; n++){
                res[m * K + k] += A[m * N + n] * B[K * n + k];
            }
        }
    }
    
}

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    int runs = (m + batch - 1) / batch;
    std::vector<float> grad (n * k);
    std::vector<float> z (batch * k);
    std::vector<float> xT(n * batch);
    for (int i = 0; i < runs; i++){
        // find batch start
        const float *x_bs = &X[i * batch * n];
        // find z (or s in lecture notes)
        matmul(x_bs, theta, z.data(), batch, n, k);
        // exp 
        for (int j = 0; j < batch * k; j++){
            z[j] = exp(z[j]);
        }
        // softmax NOTE: at this point we have batch x k
        for (int j = 0; j < batch; j++){
            float row_sum = 0.0f;
            for (int l = 0; l < k; l++){
                row_sum += z[j * k + l];
            }
            for (int l = 0; l < k; l++){
                z[j * k + l] /= row_sum;
            }
        }
        // take away I_y
        for (int j = 0; j < batch; j++){
            z[j * k + y[i * batch + j]] -= 1.0f;
        }
        // then update the grad
        // recall it is like x.T * z
        std::fill(grad.begin(), grad.end(), 0.0f);
        for (int x = 0; x < batch; x++){
            for (int y = 0; y < n; y++) {
                // x.T 
                xT[batch * y + x] = x_bs[x * n + y];
            }
        }
        matmul(xT.data(), z.data(), grad.data(), n, batch, k);

        // update sgd
        for (int j = 0; j < n * k; j++){
            theta[j] -= lr / batch  * grad[j];
        }
    }
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
