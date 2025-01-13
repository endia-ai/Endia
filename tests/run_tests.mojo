# ===----------------------------------------------------------------------=== #
# Endia 2024
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from .test_endia import *
# from .integration_tests import *


def run_unit_tests():
    # Unit Test: Test unary ops
    run_test_to_abs()
    run_test_to_abs_grad()
    run_test_to_abs_complex()

    run_test_acos()
    run_test_acos_grad()
    # run_test_acos_complex()

    run_test_asin()
    run_test_asin_grad()
    # run_test_asin_complex()

    run_test_atan()
    run_test_atan_grad()
    # run_test_atan_complex()

    run_test_cos()
    run_test_cos_grad()
    run_test_cos_complex()

    run_test_cosh()
    run_test_cosh_grad()
    run_test_cosh_complex()

    run_test_exp()
    run_test_exp_grad()
    run_test_exp_complex()

    run_test_log()
    run_test_log_grad()
    run_test_log_complex()

    run_test_neg()
    run_test_neg_grad()
    run_test_neg_complex()

    run_test_reciprocal()
    run_test_reciprocal_grad()
    run_test_reciprocal_complex()

    run_test_relu()
    run_test_relu_grad()
    # note: relu only for real numbers

    run_test_sigmoid()
    run_test_sigmoid_grad()
    # note: sigmoid only for real numbers

    run_test_sign()
    # note: sign not differentiable
    run_test_sign_complex()

    run_test_sin()
    run_test_sin_grad()
    run_test_sin_complex()

    run_test_sqrt()
    run_test_sqrt_grad()
    run_test_sqrt_complex()

    run_test_square()
    run_test_square_grad()
    run_test_square_complex()

    run_test_tan()
    run_test_tan_grad()
    # run_test_tan_complex()

    run_test_tanh()
    run_test_tanh_grad()
    run_test_tanh_complex()

    # Unit Test: Test binary ops
    run_test_add()
    run_test_add_grad()
    run_test_add_complex()
    run_test_sub()
    run_test_sub_grad()
    run_test_sub_complex()
    run_test_mul()
    run_test_mul_grad()
    run_test_mul_complex()
    run_test_div()
    run_test_div_grad()
    run_test_div_complex()
    run_test_pow()
    run_test_pow_grad()
    run_test_pow_complex()
    run_test_matmul()
    run_test_matmul_grad()
    run_test_matmul_complex()

    # Unit Test: Test reduce ops
    run_test_reduce_add()
    run_test_reduce_add_grad()
    run_test_reduce_mul()

    run_test_mean()
    run_test_mean_grad()

    # Needs FIX: Mojo breaks when using the name 'var' from pytorch
    # run_test_variance()
    # run_test_variance_grad()

    run_test_std()
    run_test_std_grad()

    run_test_reduce_max()

    run_test_reduce_argmax()

    run_test_reduce_min()

    run_test_reduce_argmin()

    # Unit Test: Test view ops
    run_test_expand()
    run_test_expand_grad()
    run_test_permute()
    run_test_permute_grad()
    run_test_squeeze()
    run_test_squeeze_grad()
    run_test_as_strided()
    run_test_as_strided_grad()
    run_test_reshape()
    run_test_reshape_grad()

    # Unit Tests: Test comparison ops
    run_test_greater_equal()
    run_test_greater()
    run_test_less_equal()
    run_test_less()

    # Unit Test: Test spacial ops
    run_test_conv1d()
    run_test_conv2d()
    run_test_conv3d()

    run_test_max_pool1d()
    run_test_max_pool2d()
    run_test_max_pool3d()

    run_test_avg_pool1d()
    run_test_avg_pool2d()
    run_test_avg_pool3d()


# def run_integration_tests():
#     # integration Tests: Test random functions
#     run_test_foo()
#     run_test_foo_grad()


def run_fft_tests():
    # FFT Tests
    fft_test()
    fft_grad_test()
    fft2_test()
    fft2_grad_test()
    fftn_test()
    fftn_grad_test()
    ifft_test()
    ifft_grad_test()
    ifft2_test()
    ifft2_grad_test()
    ifftn_test()
    ifftn_grad_test()


def run_tests():
    """
    This is the main function that runs all the tests and benchmarks.
    """
    # check if PyTorch is installed
    try:
        _ = Python.import_module("torch")

    except:
        raise "\033[91m[ERROR]\033[0m PyTorch not found. Skipping tests.\nIn order to use PyTorch please install it via:\n\033[92m magic project channel add \"pytorch\" && magic add pytorch \033[0m"

    # run tests
    run_fft_tests()
    run_unit_tests()