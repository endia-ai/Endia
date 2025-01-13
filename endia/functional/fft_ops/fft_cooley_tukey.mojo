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

import endia as nd
from endia.utils import compute_stride
from endia.functional._utils import is_contiguous, contiguous
from algorithm import parallelize, vectorize
from memory import memcpy
from sys import num_physical_cores
from endia.utils import copy_shape, NA, list_to_array_shape, array_shape_to_list
from .utils import *

alias pi = Float64(3.141592653589793)  # Maximum useful precision for Float64


fn cooley_tukey_non_recursive(
    n: Int, max_depth: Int, res_data: UnsafePointer[Scalar[DType.float64]]
):
    """
    Non-recursive Cooley-Tukey FFT splitting of the input data with a limited depth.
    """

    # Non-recursive Cooley-Tukey FFT splitting of the input data
    for iteration in range(max_depth):
        var subarray_size = n // (2**iteration)
        var num_subarrays = 2**iteration
        var temp = UnsafePointer[Scalar[DType.float64]].alloc(subarray_size * 2)

        # Split each subarray into even and odd indices
        for subarray in range(num_subarrays):
            var start = subarray * subarray_size
            var end = start + subarray_size
            var even_index = 0
            var odd_index = subarray_size // 2

            for i in range(start, end):
                if (i - start) % 2 == 0:
                    # temp.store(
                    #     2 * even_index, res_data.load[width=2](2 * i)
                    # )
                    temp.offset(2 * even_index).strided_store(
                        res_data.load[width=2](2 * i), 
                        stride=1
                    )
                    even_index += 1
                else:
                    # temp.store(
                    #     2 * odd_index, res_data.load[width=2](2 * i)
                    # )
                    temp.offset(2 * odd_index).strided_store(
                        res_data.load[width=2](2 * i), 
                        stride=1
                    )
                    odd_index += 1

            for i in range(subarray_size):
                res_data.store(2 * (start + i), temp[2 * i])
                res_data.store(2 * (start + i) + 1, temp[2 * i + 1])

        # Free the temporary buffer
        temp.free()


fn cooley_tukey_sequencial_recombine(
    n: Int, start_depth: Int, res_data: UnsafePointer[Scalar[DType.float64]]
):
    """
    Non-recursive Cooley-Tukey FFT recombination of the subsolutions. The recombination starts at a given depth.
    """
    # Allocate temporary buffers
    var temp = UnsafePointer[Scalar[DType.float64]].alloc(n * 2)
    var even = UnsafePointer[Scalar[DType.float64]].alloc(n)
    var odd = UnsafePointer[Scalar[DType.float64]].alloc(n)
    var T = UnsafePointer[Scalar[DType.float64]].alloc(n)
    var twiddle_factors = UnsafePointer[Scalar[DType.float64]].alloc(n)

    # Non-recursive Cooley-Tukey FFT recombination of the subsolutions
    for iteration in range(start_depth - 1, -1, -1):
        var subarray_size = n // (2**iteration)
        var num_subarrays = 2**iteration

        for k in range(subarray_size // 2):
            var p = (-2 * pi / subarray_size) * k
            twiddle_factors.store(
                2 * k, SIMD[DType.float64, 2](math.cos(p), math.sin(p))
            )

        for subarray in range(num_subarrays):
            var start = subarray * subarray_size
            var mid = start + subarray_size // 2

            for k in range(subarray_size // 2):
                var k_times_2 = 2 * k
                var k_times_2_plus_1 = k_times_2 + 1
                even.store(
                    k_times_2, res_data.load[width=2](2 * (start + k))
                )
                odd.store(
                    k_times_2, res_data.load[width=2](2 * (mid + k))
                )
                var twiddle_factor = twiddle_factors.load[width=2](k_times_2)
                T.store(
                    k_times_2,
                    twiddle_factor[0] * odd[k_times_2]
                    - twiddle_factor[1] * odd[k_times_2_plus_1],
                )
                T.store(
                    k_times_2_plus_1,
                    twiddle_factor[0] * odd[k_times_2_plus_1]
                    + twiddle_factor[1] * odd[k_times_2],
                )
                temp.store(
                    2 * (start + k),
                    even.load[width=2](k_times_2) + T.load[width=2](k_times_2),
                )
                temp.store(
                    2 * (mid + k),
                    even.load[width=2](k_times_2) - T.load[width=2](k_times_2),
                )

        memcpy(res_data, temp, n * 2)

    # Free the temporary buffers
    even.free()
    odd.free()
    T.free()
    twiddle_factors.free()
    temp.free()


fn fft_cooley_tukey_inplace_bit_reversal(
    workload: Int,
    data: UnsafePointer[Scalar[DType.float64]],
    reordered_arr_data: UnsafePointer[Scalar[DType.uint32]],
    dims: List[Int] = List[Int](),
):
    """
    Iterative fast Fourier transform using the Cooley-Tukey algorithm with bit-reversal permutation inplace.
    """
    # permute x according to the bit-reversal permutation
    for i in range(workload):
        var j = int(reordered_arr_data.load(i))
        if i < j:
            var tmp = data.load[width=2](2 * i)
            data.store(2 * i, data.load[width=2](2 * j))
            data.store(2 * j, tmp)

    # Cooley-Tukey FFT
    var m = 2
    while m <= workload:
        var u = SIMD[DType.float64, 2](1.0, 0.0)
        var angle = -2 * pi / m
        var w_real = math.cos(angle)
        var w_imag = math.sin(angle)

        for k in range(0, m // 2):
            for j in range(k, workload, m):
                var j_2 = 2 * j
                var j_2_plus_m = j_2 + m

                var z = data.load[width=2](j_2)
                var d = data.load[width=2](j_2_plus_m)
                var t = SIMD[DType.float64, 2](
                    u[0] * d[0] - u[1] * d[1], u[0] * d[1] + u[1] * d[0]
                )
                data.store(j_2_plus_m, z - t)
                data.store(j_2, z + t)

            # Update u for the next iteration
            u = SIMD[DType.float64, 2](
                u[0] * w_real - u[1] * w_imag, u[0] * w_imag + u[1] * w_real
            )

        m *= 2


fn fft_cooley_tukey_parallel(
    input: Array,
    dims: List[Int],
    norm: String,
    out: Optional[Array] = None,
    conj_input: Bool = False,
    conj_output: Bool = False,
    input_divisor: Float64 = 1.0,
    output_divisor: SIMD[dtype, 1] = 1.0,
) raises -> Array:
    """Compute the n-dimensional FFT using a hybrid version of the Cooley-Tukey algorithm.

    Args:
        input: The input array.
        dims: The dimensions along which to compute the FFT.
        norm: The normalization mode.
        out: The output array (optional).
        conj_input: Whether to conjugate the input data.
        conj_output: Whether to conjugate the output data.
        input_divisor: The divisor for the input data.
        output_divisor: The divisor for the output data.

    Returns:
        The n-dimensional FFT/IFFT of the input array.
    """
    var x: Array
    if not input.is_complex():
        x = complex(input, zeros_like(input))
    else:
        x = contiguous(input)

    if norm == "backward":
        pass
    elif norm == "forward":
        x = x / x.size()
    elif "ortho":
        x = x / math.sqrt(x.size())
    else:
        raise "fftn: Invalid norm"

    # setup params
    var size = x.size()
    var shape = x.shape()
    var ndim = x.ndim()
    var axes = List[Int]()
    for i in range(ndim):
        axes.append(i)
    var fft_dims = dims if dims.size > 0 else axes
    if dims.size > 0:
        for i in range(fft_dims.size):
            if fft_dims[i] < 0:
                fft_dims[i] = ndim + fft_dims[i]
            if fft_dims[i] < 0 or fft_dims[i] >= ndim:
                raise "Invalid dimension"
            if (shape[fft_dims[i]] & (shape[fft_dims[i]] - 1)) != 0:
                print(shape[fft_dims[i]])
                raise "Dimension must be a power of two"

    var res_data = UnsafePointer[Scalar[DType.float64]].alloc(size * 2)
    var data = UnsafePointer[Scalar[DType.float64]].alloc(size * 2)
    copy_complex_and_cast(data, x.data(), size, conj_input, input_divisor)

    for dim in fft_dims:
        var divisions = size // shape[dim[]]
        # var parallelize_threshold = 2**14
        var num_workers = determine_num_workers(size)
        var workload = get_workload(size, divisions, num_workers)
        var h = (
            int(math.log2(Float32(size // workload)))
        ) if divisions == 1 else 0
        var number_subtasks = num_workers if divisions == 1 else divisions

        if dim[] != ndim - 1:
            x.stride_(list_swap(x.stride(), dim[], ndim - 1))
            x.shape_(list_swap(x.shape(), dim[], ndim - 1))

        if not is_contiguous(x.array_shape()):
            execute_copy_raw(data, res_data, x.array_shape(), x.is_complex())
        else:
            var tmp = data
            data = res_data
            res_data = tmp

        # Split the data into individual subarrays to perform #workload indipendent FFTs
        if h > 0:
            cooley_tukey_non_recursive(size, h, res_data)

        # Perform the Cooley-Tukey FFT on the subarrays in parallel
        var reordered_arr_data = UnsafePointer[Scalar[DType.uint32]].alloc(
            workload
        )
        bit_reversal(workload, reordered_arr_data)

        @parameter
        fn perform_cooley_tukey_sequencial(i: Int) capturing:
            fft_cooley_tukey_inplace_bit_reversal(
                workload, res_data.offset(2 * i * workload), reordered_arr_data
            )

        parallelize[perform_cooley_tukey_sequencial](
            number_subtasks, num_workers
        )
        _ = workload, divisions
        reordered_arr_data.free()

        # Recombine the solutions of the subarrays
        if h > 0:
            cooley_tukey_sequencial_recombine(size, h, res_data)

        # Swap the data pointers
        var tmp = data
        data = res_data
        res_data = tmp

        # Swap the strides and shapes
        x.stride_(compute_stride(x.shape()))
        if dim[] != ndim - 1:
            x.stride_(list_swap(x.stride(), dim[], ndim - 1))
            x.shape_(list_swap(x.shape(), dim[], ndim - 1))

    res_data.free()

    # Copy the data back to the output array
    var output: Array
    if out:
        output = out.value()
    else:
        output = Array(x.shape(), is_complex=True)

    copy_complex_and_cast(
        output.data(), data, size, conj_output, output_divisor
    )
    data.free()

    return output


fn fft_cooley_tukey_parallel_inplace(
    input: Array,
    inout out: Array,
    dims: List[Int],
    norm: String,
    conj_input: Bool = False,
    conj_output: Bool = False,
    input_divisor: Float64 = 1.0,
    output_divisor: SIMD[dtype, 1] = 1.0,
) raises:
    """Compute the n-dimensional FFT inplace using a hybrid version of the Cooley-Tukey algorithm.

    Args:
        input: The input array.
        out: The output array.
        dims: The dimensions along which to compute the FFT.
        norm: The normalization mode.
        conj_input: Whether to conjugate the input data.
        conj_output: Whether to conjugate the output data.
        input_divisor: The divisor for the input data.
        output_divisor: The divisor for the output data.
    """
    _ = fft_cooley_tukey_parallel(
        input,
        dims,
        norm,
        out,
        conj_input,
        conj_output,
        input_divisor,
        output_divisor,
    )
