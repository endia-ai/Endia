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

from endia.utils.aliases import dtype, nelts
from endia.utils import (
    ArrayShape,
    ShapeNode,
    extract_array,
    zero_grad_rec,
    reset_node_id_recursive,
    InplaceInfo,
    build_out_string,
    compute_shape,
)
from endia.compile import FxGraph
from endia.functional import *
from endia.functional._utils import execute_copy_raw

from memory import ArcPointer, memset_zero
from algorithm import vectorize, parallelize
from time import perf_counter
from random import seed, random_ui64
from memory import UnsafePointer
import math
from python import Python, PythonObject
from collections import Optional


@value
struct Node(CollectionElement):
    """
    Node is the central data structure representing an array in the autograd engine. It is responsible for encapsulating
    all the necessary information and metadata related to an array, including its shape, data, operations, gradients, and
    dependencies.
    """

    var id: Int
    var name: String
    var shape: ArcPointer[ShapeNode]
    var data: UnsafePointer[Scalar[dtype]]
    var is_view: Bool
    var base: List[ArcPointer[Self]]
    var args: List[ArcPointer[Self]]
    var kwargs: List[ArcPointer[Self]]
    var grads: List[ArcPointer[Self]]
    var fwd: fn (inout Array, List[Array]) raises -> None
    var uew: List[
        fn (
            SIMD[dtype, nelts[dtype]() * 2 // 2],
            SIMD[dtype, nelts[dtype]() * 2 // 2],
        ) -> Tuple[
            SIMD[dtype, nelts[dtype]() * 2 // 2],
            SIMD[dtype, nelts[dtype]() * 2 // 2],
        ]
    ]
    var bew: List[
        fn (
            SIMD[dtype, nelts[dtype]() * 2 // 2],
            SIMD[dtype, nelts[dtype]() * 2 // 2],
            SIMD[dtype, nelts[dtype]() * 2 // 2],
            SIMD[dtype, nelts[dtype]() * 2 // 2],
        ) -> Tuple[
            SIMD[dtype, nelts[dtype]() * 2 // 2],
            SIMD[dtype, nelts[dtype]() * 2 // 2],
        ]
    ]
    var simd_op_list: List[
        fn (
            SIMD[dtype, nelts[dtype]() * 2 // 2],
            SIMD[dtype, nelts[dtype]() * 2 // 2],
        ) -> Tuple[
            SIMD[dtype, nelts[dtype]() * 2 // 2],
            SIMD[dtype, nelts[dtype]() * 2 // 2],
        ]
    ]
    var inplace_infos: List[InplaceInfo]
    var jvp: fn (List[Array], List[Array]) raises -> Array
    var vjp: fn (List[Array], Array, Array) raises -> List[Array]
    var requires_grad: Bool
    var compute_jvp: Bool
    var graph: Optional[ArcPointer[FxGraph]]
    var id_in_graph: Optional[Int]
    var has_real: Bool
    var has_imag: Bool
    var meta_data: ArcPointer[
        List[Int]
    ]  # some additional information encoded as a list of integers

    fn __init__(
        inout self,
        array_shape: ArrayShape,
        requires_grad: Bool = False,
        is_complex: Bool = False,
        is_view: Bool = False,
    ):
        self.id = -1
        self.name = "arg"
        self.shape = array_shape.shape_node
        if not is_view:
            var true_size = array_shape.size() if not is_complex else 2 * array_shape.size()
            self.data = UnsafePointer[Scalar[dtype]].alloc(true_size)
            memset_zero(self.data, true_size)
        else:
            self.data = UnsafePointer[Scalar[dtype]].alloc(0)
        self.is_view = is_view
        self.base = List[ArcPointer[Node]]()
        self.args = List[ArcPointer[Self]]()
        self.kwargs = List[ArcPointer[Self]]()
        self.grads = List[ArcPointer[Self]]()
        self.fwd = default_fwd
        self.uew = List[
            fn (
                SIMD[dtype, nelts[dtype]() * 2 // 2],
                SIMD[dtype, nelts[dtype]() * 2 // 2],
            ) -> Tuple[
                SIMD[dtype, nelts[dtype]() * 2 // 2],
                SIMD[dtype, nelts[dtype]() * 2 // 2],
            ]
        ]()
        self.bew = List[
            fn (
                SIMD[dtype, nelts[dtype]() * 2 // 2],
                SIMD[dtype, nelts[dtype]() * 2 // 2],
                SIMD[dtype, nelts[dtype]() * 2 // 2],
                SIMD[dtype, nelts[dtype]() * 2 // 2],
            ) -> Tuple[
                SIMD[dtype, nelts[dtype]() * 2 // 2],
                SIMD[dtype, nelts[dtype]() * 2 // 2],
            ]
        ]()
        self.inplace_infos = List[InplaceInfo]()
        self.jvp = default_jvp
        self.vjp = default_vjp
        self.simd_op_list = List[
            fn (
                SIMD[dtype, nelts[dtype]() * 2 // 2],
                SIMD[dtype, nelts[dtype]() * 2 // 2],
            ) -> Tuple[
                SIMD[dtype, nelts[dtype]() * 2 // 2],
                SIMD[dtype, nelts[dtype]() * 2 // 2],
            ]
        ]()
        self.requires_grad = requires_grad
        self.compute_jvp = False
        self.graph = None
        self.id_in_graph = None
        self.has_real = True
        self.has_imag = is_complex
        self.meta_data = ArcPointer(List[Int]())

    fn __del__(owned self):
        # print("Node __del__")
        self.data.free()

