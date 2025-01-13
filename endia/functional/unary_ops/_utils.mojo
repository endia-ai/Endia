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
    copy_shape,
)
from endia.utils import NA


trait DifferentiableUnaryOp:
    @staticmethod
    fn fwd(arg0: Array) raises -> Array:
        ...

    @staticmethod
    fn jvp(primals: List[Array], tangents: List[Array]) raises -> Array:
        ...

    @staticmethod
    fn vjp(primals: List[Array], grad: Array, out: Array) raises -> List[Array]:
        ...

    @staticmethod
    fn unary_simd_op(
        arg0_real: SIMD[dtype, nelts[dtype]() * 2 // 2],
        arg0_imag: SIMD[dtype, nelts[dtype]() * 2 // 2],
    ) -> Tuple[
        SIMD[dtype, nelts[dtype]() * 2 // 2],
        SIMD[dtype, nelts[dtype]() * 2 // 2],
    ]:
        ...

    @staticmethod
    fn __call__(inout curr: Array, args: List[Array]) raises:
        ...


fn unary_op_array(
    arg0: Array,
    name: String,
    fwd: fn (inout Array, List[Array]) raises -> None,
    jvp: fn (List[Array], List[Array]) raises -> Array,
    vjp: fn (List[Array], Array, Array) raises -> List[Array],
    inplace_op: Optional[
        fn (
            SIMD[dtype, nelts[dtype]() * 2 // 2],
            SIMD[dtype, nelts[dtype]() * 2 // 2],
        ) -> Tuple[
            SIMD[dtype, nelts[dtype]() * 2 // 2],
            SIMD[dtype, nelts[dtype]() * 2 // 2],
        ]
    ] = None,
) raises -> Array:
    var arr_shape = setup_array_shape(
        arg0.array_shape(),
        "copy_shape",
        copy_shape,
    )

    return op_array(arr_shape, arg0, NA, name, fwd, jvp, vjp, False, inplace_op)


fn execute_unary_op(inout curr: Array, args: List[Array]) raises:
    var simd_op = curr.uew()[0]
    var arg0 = contiguous(args[0])
    var arg0_data = arg0.data()
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
            var res_deinterleaved = simd_op(data0[0], data0[1])
            var res = res_deinterleaved[0].interleave(res_deinterleaved[1])
            # curr_data.store[width = 2 * ((nelts[dtype]() * 2) // 2)](
            #     idx_real, res
            # )
            curr_data.offset(idx_real).strided_store[width = 2 * ((nelts[dtype]() * 2) // 2)](
                res,
                stride=1
            )   
        if rest_size != 0:
            var rest_simd0_real = SIMD[dtype, nelts[dtype]() * 2 // 2]()
            var rest_simd0_imag = SIMD[dtype, nelts[dtype]() * 2 // 2]()

            for i in range(rest_size):
                var idx_real = (end + i) * 2
                var idx_imag = idx_real + 1
                rest_simd0_real[i] = arg0_data.load(idx_real)
                rest_simd0_imag[i] = arg0_data.load(idx_imag)
            var res = simd_op(
                rest_simd0_real,
                rest_simd0_imag,
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
                SIMD[dtype, nelts[dtype]() * 2 // 2](0),
            )[0]
            # curr_data.store[width = nelts[dtype]() * 2 // 2](i, res)
            curr_data.offset(i).strided_store[width = nelts[dtype]() * 2 // 2](res, stride=1)

        # perf_counter we vectorize along the last dimesion
        if rest_size != 0:
            var rest_simd0 = SIMD[dtype, nelts[dtype]() * 2 // 2]()
            for i in range(rest_size):
                rest_simd0[i] = arg0_data.load(i + end)
            var res = simd_op(
                rest_simd0, SIMD[dtype, nelts[dtype]() * 2 // 2]()
            )[0]
            for i in range(end, curr.size()):
                curr_data.store(i, res[i - end])

    _ = arg0
