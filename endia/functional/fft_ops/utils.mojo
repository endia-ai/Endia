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
import math
from algorithm import parallelize, vectorize
from memory import memcpy
from sys import num_physical_cores
from endia.utils import copy_shape, NA, list_to_array_shape, array_shape_to_list

alias pi = Float64(3.141592653589793)  # Maximum useful precision for Float64


fn reverse_bits_simd(
    x: SIMD[DType.uint32, nd.nelts[DType.uint32]()]
) -> SIMD[DType.uint32, nd.nelts[DType.uint32]()]:
    """
    Reverse the bits of a 32-bit integer.
    """
    var y = x
    y = ((y >> 1) & 0x55555555) | ((y & 0x55555555) << 1)
    y = ((y >> 2) & 0x33333333) | ((y & 0x33333333) << 2)
    y = ((y >> 4) & 0x0F0F0F0F) | ((y & 0x0F0F0F0F) << 4)
    y = ((y >> 8) & 0x00FF00FF) | ((y & 0x00FF00FF) << 8)
    return (y >> 16) | (y << 16)


@always_inline
fn bit_reversal(
    n: Int, reordered_arr_data: UnsafePointer[Scalar[DType.uint32]]
):
    """
    Generate a bit reversal permutation for integers from 0 to n-1.
    Works for any positive integer n.
    """
    var log2_n = int(math.ceil(math.log2(Float32(n))))
    var n_pow2 = 1 << log2_n

    var u = SIMD[DType.uint32, nd.nelts[DType.uint32]()]()
    for i in range(nd.nelts[DType.uint32]()):
        u[i] = SIMD[DType.uint32, 1](i)

    for i in range(0, n_pow2, nd.nelts[DType.uint32]()):
        var x = u + i
        var reversed = reverse_bits_simd(x) >> (32 - log2_n)

        # Only store if within original n
        if i < n:
            var to_store = reversed.cast[DType.uint32]()
            if i + 15 < n:
                reordered_arr_data.store(
                    i, to_store
                )
            else:
                for j in range(nd.nelts[DType.uint32]()):
                    if i + j < n:
                        reordered_arr_data.store(i + j, to_store[j])


fn copy_complex_and_cast[
    dst_type: DType, src_type: DType
](
    dst: UnsafePointer[Scalar[dst_type]],
    src: UnsafePointer[Scalar[src_type]],
    size: Int,
    conjugate_and_divide: Bool = False,
    divisor: SIMD[dst_type, 1] = 1,
):
    """
    Copy complex data from one buffer to another and cast the data to a different type. Optionally conjugate and divide by a scalar (usefule for inverse FFT).
    """
    if conjugate_and_divide:

        @parameter
        fn do_copy_with_div[simd_width: Int](i: Int):
            dst.store(
                2 * i,
                src.load[width = 2 * simd_width](2 * i).cast[dst_type]()
                / divisor,
            )

        vectorize[do_copy_with_div, nelts[dtype]()](size)

        for i in range(size):
            dst.store(2 * i + 1, -dst.load(2 * i + 1))
    else:

        @parameter
        fn do_copy[simd_width: Int](i: Int):
            dst.store(
                2 * i, src.load[width = 2 * simd_width](2 * i).cast[dst_type]()
            )

        vectorize[do_copy, nelts[dtype]()](size)


fn get_workload(n: Int, divisions: Int, num_workers: Int) raises -> Int:
    """
    Calculate the workload size for each worker.
    """
    var workload = (n // num_workers) if (divisions == 1) else (n // divisions)
    if (workload & (workload - 1)) != 0:
        raise "Workload size must be a power of two"
    return workload


fn list_swap(arg: List[Int], i: Int, j: Int) raises -> List[Int]:
    if i < 0 or j < 0 or i >= arg.size or j >= arg.size:
        raise "Invalid index"
    var arr = arg
    var tmp = arr[i]
    arr[i] = arr[j]
    arr[j] = tmp
    return arr


fn determine_num_workers(size: Int) raises -> Int:
    """
    Determine the number of workers to use for parallelization.
    """
    if size < 2**14:
        return 1
    elif size < 2**16:
        return 2
    elif size < 2**18:
        return 4
    else:
        return num_physical_cores()


trait DifferentiableFftOp:
    @staticmethod
    fn fwd(
        arg0: Array,
        dims: List[Int],
        norm: String,
    ) raises -> Array:
        ...

    @staticmethod
    fn jvp(primals: List[Array], tangents: List[Array]) raises -> Array:
        ...

    @staticmethod
    fn vjp(primals: List[Array], grad: Array, out: Array) raises -> List[Array]:
        ...

    @staticmethod
    fn __call__(inout curr: Array, args: List[Array]) raises:
        ...


fn fft_op_array(
    arg0: Array,
    name: String,
    fwd: fn (inout Array, List[Array]) raises -> None,
    jvp: fn (List[Array], List[Array]) raises -> Array,
    vjp: fn (List[Array], Array, Array) raises -> List[Array],
    dims: List[Int],
    norm: String,
) raises -> Array:
    var arr_shape = setup_array_shape(
        arg0.array_shape(),
        "copy_shape",
        copy_shape,
    )

    return op_array(
        arr_shape,
        arg0,
        NA,
        name,
        fwd,
        jvp,
        vjp,
        meta_data=encode_fft_params(dims, norm),
        is_complex_p=True,
    )


fn encode_fft_params(
    dims: List[Int],
    norm: String,
) raises -> List[Int]:
    var num_dims = len(dims)
    var params = List[Int](num_dims)
    params.extend(dims)
    if norm == "backward":
        params.append(0)
    elif norm == "forward":
        params.append(1)
    elif norm == "ortho":
        params.append(2)
    else:
        raise "encode: Invalid norm"
    return params


fn get_dims_from_encoded_params(
    params: List[Int],
) raises -> List[Int]:
    var num_dims = params[0]
    var dims = List[Int]()
    for i in range(1, num_dims + 1):
        dims.append(params[i])
    return dims


fn get_norm_from_encoded_params(
    params: List[Int],
) raises -> String:
    var num_dims = params[0]
    if params[num_dims + 1] == 0:
        return "backward"
    elif params[num_dims + 1] == 1:
        return "forward"
    elif params[num_dims + 1] == 2:
        return "ortho"
    else:
        raise "get_norm: Invalid norm"
