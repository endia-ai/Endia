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

from endia import Array
from endia.functional._utils import (
    contiguous,
    op_array,
    setup_array_shape,
)
from endia.utils import NA, broadcast_shapes


trait DifferentiableBinaryOp:
    """
    Trait for binary operations that are differentiable. That mean they define methods for both forward and reverse mode automatic differentiation.
    """

    @staticmethod
    fn fwd(arg0: Array, arg1: Array) raises -> Array:
        ...

    @staticmethod
    fn jvp(primals: List[Array], tangents: List[Array]) raises -> Array:
        ...

    @staticmethod
    fn vjp(primals: List[Array], grad: Array, out: Array) raises -> List[Array]:
        ...

    @staticmethod
    fn binary_simd_op(
        arg0_real: SIMD[dtype, nelts[dtype]() * 2 // 2],
        arg1_real: SIMD[dtype, nelts[dtype]() * 2 // 2],
        arg0_imag: SIMD[dtype, nelts[dtype]() * 2 // 2],
        arg1_imag: SIMD[dtype, nelts[dtype]() * 2 // 2],
    ) -> Tuple[
        SIMD[dtype, nelts[dtype]() * 2 // 2],
        SIMD[dtype, nelts[dtype]() * 2 // 2],
    ]:
        ...

    @staticmethod
    fn __call__(inout curr: Array, args: List[Array]) raises:
        ...


fn binary_op_array(
    arg0: Array,
    arg1: Array,
    name: String,
    fwd: fn (inout Array, List[Array]) raises -> None,
    jvp: fn (List[Array], List[Array]) raises -> Array,
    vjp: fn (List[Array], Array, Array) raises -> List[Array],
    inplace_op: Optional[
        fn (
            SIMD[dtype, nelts[dtype]() * 2 // 2],
            SIMD[dtype, nelts[dtype]() * 2 // 2],
            SIMD[dtype, nelts[dtype]() * 2 // 2],
            SIMD[dtype, nelts[dtype]() * 2 // 2],
        ) -> Tuple[
            SIMD[dtype, nelts[dtype]() * 2 // 2],
            SIMD[dtype, nelts[dtype]() * 2 // 2],
        ]
    ] = None,
) raises -> Array:
    var arr_shape = setup_array_shape(
        List(arg0.array_shape(), arg1.array_shape()),
        "broadcast shapes",
        broadcast_shapes,
    )
    if not arg0.has_fxgraph() and not arg1.has_fxgraph():
        compute_shape(arr_shape, arg0.requires_grad() or arg1.requires_grad())
    var args = List(
        expand(arg0, arr_shape),
        expand(arg1, arr_shape),
    )

    return op_array(
        arr_shape, args, NA, name, fwd, jvp, vjp, False, None, inplace_op
    )


fn execute_binary_op(inout curr: Array, args: List[Array]) raises:
    var simd_op = curr.bew()[0]
    var arg0 = contiguous(args[0])
    var arg1 = contiguous(args[1])
    if arg0.ndim() == 1 and arg0.shape()[0] == 1:
        arg0 = full(arg1.shape()[-1], arg0.load(0))
    if arg1.ndim() == 1 and arg1.shape()[0] == 1:
        arg1 = full(arg0.shape()[-1], arg1.load(0))
    var arg0_data = arg0.data()
    var arg1_data = arg1.data()
    var curr_data = curr.data()
    var rest_size = curr.size() % nelts[dtype]()
    var end = curr.size() - rest_size

    if curr.is_complex():
        for i in range(0, end, nelts[dtype]()):
            var idx_real = i * 2
            # var idx_imag = idx_real + 1
            var data0 = arg0_data.load[width = nelts[dtype]() * 2](
                idx_real
            ).deinterleave()
            var data1 = arg1_data.load[width = nelts[dtype]() * 2](
                idx_real
            ).deinterleave()
            var res_deinterleaved = simd_op(
                data0[0], data1[0], data0[1], data1[1]
            )
            var res = res_deinterleaved[0].interleave(res_deinterleaved[1])
            # curr_data.store[width = 2 * ((nelts[dtype]() * 2) // 2)](
            #     idx_real, res
            # )
            curr_data.offset(idx_real).strided_store[width = 2 * ((nelts[dtype]() * 2) // 2)](
                res,
                stride=1,
            )
        if rest_size != 0:
            var rest_simd0_real = SIMD[dtype, nelts[dtype]() * 2 // 2]()
            var rest_simd0_imag = SIMD[dtype, nelts[dtype]() * 2 // 2]()
            var rest_simd1_real = SIMD[dtype, nelts[dtype]() * 2 // 2]()
            var rest_simd1_imag = SIMD[dtype, nelts[dtype]() * 2 // 2]()

            for i in range(rest_size):
                var idx_real = (end + i) * 2
                var idx_imag = idx_real + 1
                rest_simd0_real[i] = arg0_data.load(idx_real)
                rest_simd0_imag[i] = arg0_data.load(idx_imag)
                rest_simd1_real[i] = arg1_data.load(idx_real)
                rest_simd1_imag[i] = arg1_data.load(idx_imag)
            var res = simd_op(
                rest_simd0_real,
                rest_simd1_real,
                rest_simd0_imag,
                rest_simd1_imag,
            )
            for i in range(rest_size):
                var idx_real = (end + i) * 2
                var idx_imag = idx_real + 1
                curr_data.store(idx_real, res[0][i])
                curr_data.store(idx_imag, res[1][i])
    else:
        for i in range(0, end, nelts[dtype]()):
            var res = simd_op(
                arg0_data.load[width = nelts[dtype]() * 2 // 2](i),
                arg1_data.load[width = nelts[dtype]() * 2 // 2](i),
                SIMD[dtype, nelts[dtype]() * 2 // 2](0),
                SIMD[dtype, nelts[dtype]() * 2 // 2](0),
            )[0]
            # curr_data.store[width = nelts[dtype]() * 2 // 2](i, res)
            curr_data.offset(i).strided_store[width = nelts[dtype]() * 2 // 2](
                res, stride=1
            )

        # perf_counter we vectorize along teh last dimesion
        if rest_size != 0:
            var rest_simd0 = SIMD[dtype, nelts[dtype]() * 2 // 2]()
            var rest_simd1 = SIMD[dtype, nelts[dtype]() * 2 // 2]()
            for i in range(rest_size):
                rest_simd0[i] = arg0_data.load(i + end)
                rest_simd1[i] = arg1_data.load(i + end)
            var res = simd_op(
                rest_simd0,
                rest_simd1,
                SIMD[dtype, nelts[dtype]() * 2 // 2](0),
                SIMD[dtype, nelts[dtype]() * 2 // 2](0),
            )[0]
            for i in range(end, curr.size()):
                curr_data.store(i, res[i - end])

    _ = arg0
    _ = arg1
    _ = curr
    _ = args
