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

from .node import Node


fn default_fwd(inout curr: Array, args: List[Array]) raises -> None:
    print("Attention: Default fwd is being used!")
    pass


fn default_vjp(
    primals: List[Array], grad: Array, out: Array
) raises -> List[Array]:
    print("Attention: Default vjp is being used!")
    return grad


fn default_jvp(primals: List[Array], tangents: List[Array]) raises -> Array:
    print("Attention: Default jvp is being used!")
    return tangents[0]
    

###############################################################################################################
#                                                       Array
###############################################################################################################
@value
struct Array(CollectionElement, Stringable, Writable):
    """
    Array is the primary data structure in the autograd engine.
    It serves as a wrapper around the Node struct, which encapsulates the array's data, shape, gradients, and other metadata.
    """

    var node: ArcPointer[Node]

    fn __init__(
        inout self,
        shape: List[Int],
        requires_grad: Bool = False,
        is_complex: Bool = False,
    ):  
        self.node = ArcPointer(Node(ArrayShape(shape), requires_grad, is_complex))

    fn __init__(inout self, array_shape: ArrayShape, is_view: Bool = False):
        self.node = ArcPointer[Node](Node(ArrayShape(array_shape.shape_node), is_view=is_view))

    fn __copyinit__(inout self, other: Self):
        self.node = other.node

    fn __moveinit__(inout self, owned other: Self):
        self.node = other.node^

    fn __init__(inout self, node: ArcPointer[Node]):
        self.node = node

    fn __init__(
        inout self, input_string: String, requires_grad: Bool = False
    ) raises:
        self = extract_array(input_string)
        self.requires_grad_(requires_grad)

    fn id(self) -> Int:
        return self.node[].id

    fn id_(inout self, id: Int):
        self.node[].id = id

    fn array_shape(self) raises -> ArrayShape:
        return ArrayShape(self.node[].shape)

    fn array_shape_(inout self, shape: ArrayShape):
        self.node[].shape[].shape = shape.shape_node[].shape
        self.node[].shape[].stride = shape.shape_node[].stride
        self.node[].shape[].storage_offset = shape.shape_node[].storage_offset
        self.node[].shape[].ndim = shape.shape_node[].ndim
        self.node[].shape[].size = shape.shape_node[].size
        self.node[].shape[].is_computed = shape.shape_node[].is_computed

    fn is_computed(self) -> Bool:
        return self.node[].shape[].is_computed

    fn is_computed_(inout self, is_computed: Bool):
        self.node[].shape[].is_computed = is_computed

    fn is_graph_node_computed(self) raises -> Bool:
        if self.has_fxgraph():
            var graph = self.graph()
            var id_in_graph = self.id_in_graph()
            return graph[].trace[id_in_graph].is_computed
        return False

    fn is_graph_node_computed_(inout self, is_computed: Bool) raises:
        if self.has_fxgraph():
            var graph = self.graph()
            var id_in_graph = self.id_in_graph()
            graph[].trace[id_in_graph].is_computed = is_computed

    fn postpone_as_grpah_output(inout self) raises:
        if self.has_fxgraph():
            var graph = self.graph()
            var id_in_graph = self.id_in_graph()
            var posponed_outputs = graph[].postponed_outputs
            if not id_in_graph in posponed_outputs:
                graph[].postponed_outputs.append(id_in_graph)

    fn args(self) -> List[Array]:
        var res = List[Array]()
        for arg in self.node[].args:
            res.append(Array(arg[]))
        return res

    fn args_(inout self, args: List[Array]):
        self.node[].args.clear()
        for arg in args:
            self.node[].args.append(arg[].node)

    fn clear_args(inout self):
        self.node[].args.clear()
        self.node[].shape[].args.clear()

    fn remove_grad(inout self):
        self.node[].grads.clear()

    fn kwargs(self) -> List[Array]:
        var res = List[Array]()
        for arg in self.node[].kwargs:
            res.append(Array(arg[]))
        return res

    fn kwargs_(inout self, kwargs: List[Array]):
        for arg in kwargs:
            self.node[].kwargs.append(arg[].node)

    fn id_in_graph_(inout self, id_in_graph: Int):
        self.node[].id_in_graph = id_in_graph

    fn id_in_graph(self) -> Int:
        if self.node[].id_in_graph:
            var id = self.node[].id_in_graph
            return id.unsafe_take()
        return -1

    fn graph(self) raises -> ArcPointer[FxGraph]:
        if not self.has_fxgraph():
            raise "Error: No graph set for this node"
        var graph_opt = self.node[].graph
        return graph_opt.unsafe_take()

    fn data_(inout self, owned data_ptr: UnsafePointer[Scalar[dtype]]):
        self.node[].data.free()
        self.node[].data = data_ptr

    fn graph_(inout self, graph: ArcPointer[FxGraph]):
        self.node[].graph = graph

    fn has_fxgraph(self) -> Bool:
        return self.node[].graph and self.node[].id_in_graph

    fn is_breakpoint(self) raises -> Bool:
        if self.has_fxgraph():
            var graph = self.graph()
            var id_in_graph = self.id_in_graph()
            return graph[].trace[id_in_graph].is_breakpoint
        return False

    fn is_breakpoint_(inout self, is_breakpoint: Bool) raises:
        if self.has_fxgraph():
            var graph = self.graph()
            var id_in_graph = self.id_in_graph()
            graph[].trace[id_in_graph].is_breakpoint = is_breakpoint

    fn item(self, idx: Int) raises -> Array:
        var res = Array(1)

        if self.has_fxgraph():
            # if the curretn node points to a valid graph we are able to compute the current node with a compiled subgraph
            var graph = self.graph()
            var id_in_graph = self.id_in_graph()
            var graph_node = graph[].trace[id_in_graph]
            var array_in_graph = graph_node.array_in_graph
            array_in_graph.postpone_as_grpah_output()
            # graph[].postponed_outputs.append(id_in_graph)

            if not graph_node.is_computed:
                var array_in_graph = self.graph_dual()
                var subgraph: ArcPointer[FxSubgraph]
                if not graph_node.sub_graph:
                    var compile_with_MAX = graph[].compile_with_MAX
                    subgraph = graph[].subgraph(compile_with_MAX)
                    graph[].trace[id_in_graph].sub_graph = subgraph
                else:
                    subgraph = graph_node.subgraph()

                subgraph[].execute()
                # print("\nFxSubgraph:")
                # subgraph.print()
                res.store(0, array_in_graph.load(idx))

                graph[].trace[id_in_graph].is_computed = True
                graph[].trace[id_in_graph].is_breakpoint = True
                return res
            else:
                res.store(0, 0)
                return res

        res.store(0, self.load(idx))
        return res

    fn setup_array_shape(inout self, array_shape: ArrayShape) raises:
        self.node[].shape = array_shape.shape_node

    fn uew(
        self,
    ) -> List[
        fn (
            SIMD[dtype, nelts[dtype]() * 2 // 2],
            SIMD[dtype, nelts[dtype]() * 2 // 2],
        ) -> Tuple[
            SIMD[dtype, nelts[dtype]() * 2 // 2],
            SIMD[dtype, nelts[dtype]() * 2 // 2],
        ]
    ]:
        return self.node[].uew

    fn bew(
        self,
    ) -> List[
        fn (
            SIMD[dtype, nelts[dtype]() * 2 // 2],
            SIMD[dtype, nelts[dtype]() * 2 // 2],
            SIMD[dtype, nelts[dtype]() * 2 // 2],
            SIMD[dtype, nelts[dtype]() * 2 // 2],
        ) -> Tuple[
            SIMD[dtype, nelts[dtype]() * 2 // 2],
            SIMD[dtype, nelts[dtype]() * 2 // 2],
        ]
    ]:
        return self.node[].bew

    fn inplace_infos(self) -> List[InplaceInfo]:
        return self.node[].inplace_infos

    fn append_arg(inout self, arg: Array):
        self.node[].args.append(arg.node)

    fn append_inplace_info(inout self, inplace_info: InplaceInfo):
        self.node[].inplace_infos.append(inplace_info)

    fn append_uew(
        inout self,
        uew: fn (
            SIMD[dtype, nelts[dtype]() * 2 // 2],
            SIMD[dtype, nelts[dtype]() * 2 // 2],
        ) -> Tuple[
            SIMD[dtype, nelts[dtype]() * 2 // 2],
            SIMD[dtype, nelts[dtype]() * 2 // 2],
        ],
    ):
        self.node[].uew.append(uew)

    fn append_bew(
        inout self,
        bew: fn (
            SIMD[dtype, nelts[dtype]() * 2 // 2],
            SIMD[dtype, nelts[dtype]() * 2 // 2],
            SIMD[dtype, nelts[dtype]() * 2 // 2],
            SIMD[dtype, nelts[dtype]() * 2 // 2],
        ) -> Tuple[
            SIMD[dtype, nelts[dtype]() * 2 // 2],
            SIMD[dtype, nelts[dtype]() * 2 // 2],
        ],
    ):
        self.node[].bew.append(bew)

    fn shape(self) raises -> List[Int]:
        var array_shape = self.array_shape()
        compute_shape(array_shape, self.requires_grad() or self.has_fxgraph())
        return array_shape.shape()

    fn shape_(inout self, shape: List[Int]) raises:
        var array_shape = self.array_shape()
        compute_shape(array_shape, self.requires_grad() or self.has_fxgraph())
        array_shape.shape_node[].shape = shape

    fn stride_(inout self, stride: List[Int]) raises:
        var array_shape = self.array_shape()
        compute_shape(array_shape, self.requires_grad() or self.has_fxgraph())
        array_shape.shape_node[].stride = stride

    fn stride(self) raises -> List[Int]:
        var array_shape = self.array_shape()
        compute_shape(array_shape, self.requires_grad() or self.has_fxgraph())
        return array_shape.stride()

    fn storage_offset(self) raises -> Int:
        var array_shape = self.array_shape()
        compute_shape(array_shape, self.requires_grad() or self.has_fxgraph())
        return array_shape.storage_offset()

    fn storage_offset_(inout self, storage_offset: Int) raises:
        var array_shape = self.array_shape()
        array_shape.shape_node[].storage_offset = storage_offset

    fn ndim(self) raises -> Int:
        var array_shape = self.array_shape()
        compute_shape(array_shape, self.requires_grad() or self.has_fxgraph())
        return array_shape.ndim()

    fn size(self) raises -> Int:
        var array_shape = self.array_shape()
        compute_shape(array_shape, self.requires_grad() or self.has_fxgraph())
        return array_shape.size()

    fn is_view(self) -> Bool:
        return self.node[].is_view

    fn is_view_(inout self, is_view: Bool):
        self.node[].is_view = is_view

    fn base_(inout self, base: Array):
        self.node[].base.clear()
        self.node[].base.append(base.node)

    fn base(self) -> Array:
        if self.is_view():
            return Array(self.node[].base[0]).base()
        return self

    fn requires_grad(self) -> Bool:
        return self.node[].requires_grad

    fn requires_grad_(inout self, requires_grad: Bool):
        self.node[].requires_grad = requires_grad

    fn _requires_grad(self, requires_grad: Bool) -> Self:
        var node = self.node
        node[].requires_grad = requires_grad
        return self

    fn meta_data(self) -> List[Int]:
        return self.node[].meta_data[]

    fn meta_data_(inout self, meta_data: List[Int]):
        self.node[].meta_data[] = meta_data

    fn has_real(self) -> Bool:
        return self.node[].has_real

    fn has_real_(inout self, has_real: Bool):
        self.node[].has_real = has_real

    fn has_imag(self) -> Bool:
        return self.node[].has_imag

    fn has_imag_(inout self, has_imag: Bool):
        self.node[].has_imag = has_imag

    fn is_complex(self) -> Bool:
        return self.has_real() and self.has_imag()

    fn is_complex_(inout self, is_complex: Bool):
        self.has_real_(True)
        self.has_imag_(is_complex)

    fn data(self) -> UnsafePointer[Scalar[dtype]]:
        if self.is_view():
            return self.base().node[].data
        return self.node[].data

    fn real_idx(self, idx: Int) -> Int:
        if self.is_complex():
            return idx * 2
        return idx

    fn imag_idx(self, idx: Int) -> Int:
        if self.is_complex():
            return idx * 2 + 1
        return idx

    fn load[width: Int = 1](self, idx: Int) -> SIMD[dtype, width]:
        if self.is_view():
            var nd_idx = compute_nd_index(idx, self.node[].shape[].shape)
            var base_idx = compute_storage_offset(
                nd_idx,
                self.node[].shape[].stride,
                self.node[].shape[].storage_offset,
            )
            var base = self.node[].base[0]
            return base[].data.load[width=width](self.real_idx(base_idx))
        else:
            return self.data().load[width=width](self.real_idx(idx))

    fn store[
        width: Int = 1
    ](inout self, idx: Int, data: SIMD[dtype, width]) -> None:
        if self.is_view():
            var nd_idx = compute_nd_index(idx, self.node[].shape[].shape)
            var base_idx = compute_storage_offset(
                nd_idx,
                self.node[].shape[].stride,
                self.node[].shape[].storage_offset,
            )
            var base = self.node[].base[0]
            base[].data.store(self.real_idx(base_idx), data)
            # base[].data.offset(self.real_idx(base_idx)).strided_store[width=width](data, stride=1)
        else:
            return self.data().store(self.real_idx(idx), data)
            # self.data().offset(self.real_idx(idx)).strided_store[width=width](data, stride=1)

    fn load_imag[width: Int = 1](self, idx: Int) -> SIMD[dtype, width]:
        if self.is_view():
            var nd_idx = compute_nd_index(idx, self.node[].shape[].shape)
            var base_idx = compute_storage_offset(
                nd_idx,
                self.node[].shape[].stride,
                self.node[].shape[].storage_offset,
            )
            var base = self.node[].base[0]
            return base[].data.load[width=width](self.imag_idx(base_idx))
        else:
            return self.data().load[width=width](self.imag_idx(idx))

    fn store_imag[
        width: Int = 1
    ](inout self, idx: Int, data: SIMD[dtype, width]) -> None:
        if self.is_view():
            var nd_idx = compute_nd_index(idx, self.node[].shape[].shape)
            var base_idx = compute_storage_offset(
                nd_idx,
                self.node[].shape[].stride,
                self.node[].shape[].storage_offset,
            )
            var base = self.node[].base[0]
            base[].data.offset(self.imag_idx(base_idx)).strided_store[width=width](data, stride=1)
        else:
            # return self.data().store[width=width](self.imag_idx(idx), data)
            self.data().offset(self.imag_idx(idx)).strided_store[width=width](data, stride=1)

    fn load_complex[
        width: Int = 1
    ](self, idx: Int) -> Tuple[
        SIMD[dtype, 2 * width // 2], SIMD[dtype, 2 * width // 2]
    ]:
        if self.is_view():
            var nd_idx = compute_nd_index(idx, self.node[].shape[].shape)
            var base_idx = compute_storage_offset(
                nd_idx,
                self.node[].shape[].stride,
                self.node[].shape[].storage_offset,
            )
            var base = self.node[].base[0]
            var res_deinterleaved = base[].data.load[width = 2 * width](
                self.real_idx(base_idx)
            ).deinterleave()
            return (res_deinterleaved[0], res_deinterleaved[1])
        else:
            var res_deinterleaved = self.data().load[width = 2 * width](
                self.real_idx(idx)
            ).deinterleave()
            return (res_deinterleaved[0], res_deinterleaved[1])

    fn store_complex[
        width: Int = 1
    ](
        inout self, idx: Int, real: SIMD[dtype, width], imag: SIMD[dtype, width]
    ) raises -> None:
        if self.is_view():
            var nd_idx = compute_nd_index(idx, self.shape())
            var base_idx = compute_storage_offset(
                nd_idx, self.stride(), self.storage_offset()
            )
            var base = self.node[].base[0]
            # base[].data.store[width = 2 * width](
            #     self.real_idx(base_idx), real.interleave(imag)
            # )
            base[].data.offset(self.real_idx(base_idx)).strided_store[width=2 * width](real.interleave(imag), stride=1)
        else:
            # return self.data().store[width = 2 * width](
            #     self.real_idx(idx), real.interleave(imag)
            # )
            self.data().offset(self.real_idx(idx)).strided_store[width=2 * width](real.interleave(imag), stride=1)

    fn compute_jvp(self) -> Bool:
        return self.node[].compute_jvp

    fn set_compute_jvp(inout self, compute_jvp: Bool):
        self.node[].compute_jvp = compute_jvp

    fn set_fwd(
        inout self,
        fwd: fn (inout Array, List[Array]) raises -> None,
    ):
        self.node[].fwd = fwd

    fn fwd(
        self,
    ) raises -> fn (inout Array, List[Array]) raises -> None:
        # var fwd_opt = self.node[].fwd
        # if fwd_opt:
        #     var fwd = fwd_opt.unsafe_take()
        #     return fwd
        # raise "Error: No FWD set for this node"
        return self.node[].fwd

    fn jvp_(
        inout self,
        jvp: fn (List[Array], List[Array]) raises -> Array,
    ):
        self.node[].jvp = jvp

    fn jvp(
        self,
    ) raises -> fn (List[Array], List[Array]) raises -> Array:
        # if not self.node[].jvp:
        #     raise "Error: No JVP set for this node"
        # var jvp = self.node[].jvp
        # return jvp.unsafe_take()
        return self.node[].jvp

    fn vjp_(
        inout self,
        vjp: fn (List[Array], Array, Array) raises -> List[Array],
    ):
        self.node[].vjp = vjp

    fn vjp(
        self,
    ) raises -> fn (List[Array], Array, Array) raises -> List[Array]:
        # if not self.node[].vjp:
        #     raise "Error: No VJP set for this node"
        # var vjp = self.node[].vjp
        # return vjp.unsafe_take()
        return self.node[].vjp

    fn has_grad(self) -> Bool:
        return len(self.node[].grads) > 0

    fn grad_(inout self, grad: Array):
        self.node[].grads.clear()
        self.node[].grads.append(grad.node)

    fn grad(self) raises -> Array:
        if not self.has_grad():
            return Array(self.shape())
        return Array(self.node[].grads[0])

    fn set_name(inout self, name: String):
        self.node[].name = name

    fn name(self) -> String:
        return self.node[].name

    fn execute_fwd(inout self) raises:
        var array_shape = self.array_shape()
        array_shape.execute_fwd(array_shape.args())
        # if self.node[].fwd:
        var args = self.args()
        # var kwargs = self.kwargs()
        var array_copy = Array(self.node)
        self.fwd()(array_copy, args)
        if self.compute_jvp():
            var jvp = self.jvp()
            var primals = self.args()
            var tangents = List[Array]()
            for arg in primals:
                tangents.append(arg[].grad())
            self.grad_(jvp(primals, tangents))

    fn __str__(self) -> String:
        var storage_offset = ""
        var out: String = ""
        # out += storage_offset + "Array("
        var idx = 0
        var dim = 0
        var indent = " "
        var ndim = self.node[].shape[].ndim
        if (
            ndim == 1
            and self.node[].shape[].shape[0] == 1
            and not self.is_complex()
        ):
            out = str(self.load(0))
        else:
            build_out_string(self, out, idx, dim, indent)
        # out += ", shape=("
        # var ndim = self.node[].shape[].ndim
        # var shape = self.node[].shape[].shape
        # var stride = self.node[].shape[].stride
        # for i in range(ndim):
        #     out += str(shape[i])
        #     out += ", " if i < ndim - 1 else ""
        # out += "), stride: ("
        # for i in range(ndim):
        #     out += str(stride[i])
        #     out += "x" if i < ndim - 1 else ""
        # out += "), storage_offset: " + str(self.node[].shape[].storage_offset)
        # out += ", dtype=" + str(dtype) + ")"
        return out

    fn write_to[W: Writer](self, inout writer: W):
        writer.write[String](str(self))

    fn execute_fwds(
        inout self,
    ) raises:
        if not self.node[].simd_op_list:
            return

        var ops = self.node[].simd_op_list

        var size = self.size()
        var simd_end = size - nelts[dtype]()
        var normal_start = (size // (nelts[dtype]())) * nelts[dtype]()

        for n in range(0, simd_end, nelts[dtype]()):
            var simd_args = self.load[nelts[dtype]() * 2 // 2](n)
            for op in ops:
                simd_args = op[](
                    simd_args, SIMD[dtype, nelts[dtype]() * 2 // 2]()
                )[0]
            self.store[nelts[dtype]() * 2 // 2](n, simd_args)

        var rest = SIMD[dtype, nelts[dtype]() * 2 // 2]()
        for n in range(normal_start, size):
            rest[n - normal_start] = self.load(n)
        for op in ops:
            rest = op[](rest, SIMD[dtype, nelts[dtype]() * 2 // 2]())[0]
        for n in range(normal_start, size):
            self.store(n, rest[n - normal_start])

    fn graph_dual(self) raises -> Self:
        if not self.has_fxgraph():
            raise "Error: No graph set for this node"
        var graph = self.graph()
        var graph_id = self.id_in_graph()
        return graph[].trace[graph_id].array_in_graph

    fn backward(self, create_graph: Bool = False) raises:
        backward(self, create_graph)

    fn zero_grad(inout self):
        zero_grad_rec(self)

    fn T(self) raises -> Array:
        if self.ndim() == 1:
            return self
        return permute(self, ArrayShape(List(-1, -2)))

    fn reshape(self, shape: List[Int]) raises -> Array:
        return reshape(self, shape)

    fn __getitem__(self, *slices: Slice) raises -> Array:
        var slices_list = List[Slice]()
        for i in range(len(slices)):
            slices_list.append(slices[i])
        return array_slice(self, slices_list)

    fn __setitem__(
        inout self, *slices: Slice, value: Variant[SIMD[dtype, 1], Self]
    ) raises:
        var slices_list = List[Slice]()
        for i in range(len(slices)):
            slices_list.append(slices[i])
        var subarray = array_slice(self, slices_list)
        var subarray_shape = subarray.shape()
        if value.isa[Self]():
            var value = value[Self][]
            var value_shape = value.shape()
            for i in range(len(subarray_shape)):
                if subarray_shape[i] != value_shape[i]:
                    raise "Error: Shapes do not match"

            var dst = subarray
            var src = value
            var dst_rank = subarray.ndim()
            var src_rank = value.ndim()
            var dst_shape = List[Int]()
            var src_shape = List[Int]()
            var dst_stride = List[Int]()
            var src_stride = List[Int]()
            var dst_storage_offset = dst.storage_offset()
            var src_storage_offset = src.storage_offset()

            if dst_rank == 1 and src_rank == 1:
                dst_shape.append(1)
                dst_stride.append(0)
                dst_shape.append(dst.shape()[0])
                dst_stride.append(dst.stride()[0])
                src_shape.append(1)
                src_stride.append(0)
                src_shape.append(src.shape()[0])
                src_stride.append(src.stride()[0])
            else:
                dst_shape = dst.shape()
                dst_stride = dst.stride()
                src_shape = src.shape()
                src_stride = src.stride()

            var rank = len(dst_shape)
            var rows = dst_shape[rank - 2]
            var cols = dst_shape[rank - 1]
            var size = 1
            for i in range(rank):
                size *= dst_shape[i]
            var dest_data = dst.data()
            var source_data = src.data()

            for k in range(0, size, rows * cols):
                var dst_nd_idx = compute_nd_index(k, dst_shape)
                var dst_base_idx = compute_storage_offset(
                    dst_nd_idx, dst_stride, dst_storage_offset
                )
                var src_nd_idx = compute_nd_index(k, src_shape)
                var src_base_idx = compute_storage_offset(
                    src_nd_idx, src_stride, src_storage_offset
                )

                for i in range(rows):
                    var dst_i_idx = dst_base_idx + i * dst_stride[rank - 2]
                    var src_i_idx = src_base_idx + i * src_stride[src_rank - 2]

                    if dst.is_complex() and src.is_complex():
                        if (
                            dst_stride[rank - 1] == 1
                            and src_stride[src_rank - 1] == 1
                        ):
                            alias simd_width = nelts[dtype]() * 2 // 2

                            @parameter
                            fn copy_v_complex[simd_width: Int](j: Int):
                                var dst_j_idx = 2 * (
                                    dst_i_idx + j * dst_stride[rank - 1]
                                )
                                var src_j_idx = 2 * (
                                    src_i_idx + j * src_stride[rank - 1]
                                )
                                # dest_data.store[width = 2 * simd_width](
                                #     dst_j_idx,
                                #     source_data.load[width = 2 * simd_width](
                                #         src_j_idx
                                #     ),
                                # )
                                dest_data.offset(dst_j_idx).strided_store[width=2 * simd_width](
                                    source_data.load[width=2 * simd_width](src_j_idx),
                                    stride=1
                                )

                            vectorize[copy_v_complex, 2 * nelts[dtype]()](cols)

                        else:
                            for j in range(cols):
                                var dst_j_idx = 2 * (
                                    dst_i_idx + j * dst_stride[rank - 1]
                                )
                                var src_j_idx = 2 * (
                                    src_i_idx + j * src_stride[rank - 1]
                                )
                                # dest_data.store[width=2](
                                #     dst_j_idx,
                                #     source_data.load[width=2](src_j_idx),
                                # )
                                dest_data.offset(dst_j_idx).strided_store[width=2](
                                    source_data.load[width=2](src_j_idx),
                                    stride=1
                                )

                    else:
                        if (
                            dst_stride[rank - 1] == 1
                            and src_stride[src_rank - 1] == 1
                        ):
                            alias simd_width = nelts[dtype]() * 2 // 2

                            @parameter
                            fn copy_v[simd_width: Int](j: Int):
                                var dst_j_idx = dst_i_idx + j * dst_stride[
                                    rank - 1
                                ]
                                var src_j_idx = src_i_idx + j * src_stride[
                                    src_rank - 1
                                ]
                                # dest_data.store[width=simd_width](
                                #     dst_j_idx,
                                #     source_data.load[width=simd_width](
                                #         src_j_idx
                                #     ),
                                # )
                                dest_data.offset(dst_j_idx).strided_store[width=simd_width](
                                    source_data.load[width=simd_width](src_j_idx),
                                    stride=1
                                )

                            vectorize[copy_v, nelts[dtype]()](cols)

                        else:
                            for j in range(cols):
                                var dst_j_idx = dst_i_idx + j * dst_stride[
                                    rank - 1
                                ]
                                var src_j_idx = src_i_idx + j * src_stride[
                                    rank - 1
                                ]
                                dest_data.store(
                                    dst_j_idx, source_data.load(src_j_idx)
                                )

            _ = value

        elif value.isa[SIMD[dtype, 1]]():
            for i in range(subarray.size()):
                subarray.store(i, value[SIMD[dtype, 1]])

        else:
            raise "Error: Invalid value type"

    fn __add__(self, other: Array) raises -> Array:
        return add(self, other)

    fn __add__(self, other: SIMD[dtype, 1]) raises -> Array:
        var other_array = full(self.shape()[self.ndim() - 1], other)
        return add(self, other_array)

    fn __radd__(self, other: Array) raises -> Array:
        return add(other, self)

    fn __radd__(self, other: SIMD[dtype, 1]) raises -> Array:
        var other_array = full(self.shape()[self.ndim() - 1], other)
        return add(other_array, self)

    fn __iadd__(inout self, other: Array) raises:
        var res = add(self, other)
        execute_copy_raw(
            res.data(),
            self.data(),
            self.array_shape(),
            res.is_complex(),
        )
        _ = res

    fn __iadd__(inout self, other: SIMD[dtype, 1]) raises:
        self = self.__add__(other)

    fn __sub__(self, other: Array) raises -> Array:
        return sub(self, other)

    fn __sub__(self, other: SIMD[dtype, 1]) raises -> Array:
        var other_array = full(self.shape()[self.ndim() - 1], other)
        return sub(self, other_array)

    fn __rsub__(self, other: Array) raises -> Array:
        return sub(other, self)

    fn __rsub__(self, other: SIMD[dtype, 1]) raises -> Array:
        var other_array = full(self.shape()[self.ndim() - 1], other)
        return sub(other_array, self)

    fn __isub__(inout self, other: Array) raises:
        var res = sub(self, other)
        execute_copy_raw(
            res.data(),
            self.data(),
            self.array_shape(),
            res.is_complex(),
        )
        _ = res

    fn __isub__(inout self, other: SIMD[dtype, 1]) raises:
        self = self.__sub__(other)

    fn __mul__(self, other: Array) raises -> Array:
        return mul(self, other)

    fn __mul__(self, other: SIMD[dtype, 1]) raises -> Array:
        var other_array = full(self.shape()[self.ndim() - 1], other)
        return mul(self, other_array)

    fn __rmul__(self, other: Array) raises -> Array:
        return mul(other, self)

    fn __rmul__(self, other: SIMD[dtype, 1]) raises -> Array:
        var other_array = full(self.shape()[self.ndim() - 1], other)
        return mul(other_array, self)

    fn __imul__(inout self, other: Array) raises:
        var res = mul(self, other)
        execute_copy_raw(
            res.data(),
            self.data(),
            self.array_shape(),
            res.is_complex(),
        )
        _ = res

    fn __imul__(inout self, other: SIMD[dtype, 1]) raises:
        self = self.__mul__(other)

    fn __truediv__(self, other: Array) raises -> Array:
        return div(self, other)

    fn __truediv__(self, other: SIMD[dtype, 1]) raises -> Array:
        var other_array = full(self.shape()[self.ndim() - 1], other)
        return div(self, other_array)

    fn __rtruediv__(self, other: Array) raises -> Array:
        return div(other, self)

    fn __rtruediv__(self, other: SIMD[dtype, 1]) raises -> Array:
        var other_array = full(self.shape()[self.ndim() - 1], other)
        return div(other_array, self)

    fn __itruediv__(inout self, other: Array) raises:
        var res = div(self, other)
        execute_copy_raw(
            res.data(),
            self.data(),
            self.array_shape(),
            res.is_complex(),
        )
        _ = res

    fn __itruediv__(inout self, other: SIMD[dtype, 1]) raises:
        self = self.__truediv__(other)

    fn __matmul__(self, other: Array) raises -> Array:
        return matmul(self, other)

    fn __rmatmul__(self, other: Array) raises -> Array:
        return matmul(other, self)

    fn __neg__(self) raises -> Array:
        return neg(self)

    fn __pow__(self, other: Array) raises -> Array:
        return pow_to(self, other)

    fn __pow__(self, other: SIMD[dtype, 1]) raises -> Array:
        var other_array = full(self.shape()[self.ndim() - 1], other)
        return pow_to(self, other_array)

    fn __rpow__(self, other: Array) raises -> Array:
        return pow_to(other, self)

    fn __rpow__(self, other: SIMD[dtype, 1]) raises -> Array:
        var other_array = full(self.shape()[self.ndim() - 1], other)
        return pow_to(other_array, self)

    fn __ipow__(inout self, other: Array) raises:
        var res = pow_to(self, other)
        execute_copy_raw(
            res.data(),
            self.data(),
            self.array_shape(),
            res.is_complex(),
        )
        _ = res

    fn __ipow__(inout self, other: SIMD[dtype, 1]) raises:
        self = self.__pow__(other)

    fn __eq__(self, other: Array) raises -> Bool:
        var shape = self.shape()
        var other_shape = other.shape()
        for i in range(len(shape)):
            if shape[i] != other_shape[i]:
                return False
        var eq_compared = equal(self, other)
        if prod(eq_compared).load(0) == 1:
            return True
        return False

    fn __eq__(self, other: SIMD[dtype, 1]) raises -> Bool:
        var other_array = full(self.shape()[self.ndim() - 1], other)
        return self.__eq__(other_array)

    fn __ne__(self, other: Array) raises -> Bool:
        return not self.__eq__(other)

    fn __ne__(self, other: SIMD[dtype, 1]) raises -> Bool:
        var other_array = full(self.shape()[self.ndim() - 1], other)
        return self.__ne__(other_array)

    fn __ge__(self, other: Array) raises -> Bool:
        var ge_compared = greater_equal(self, other)
        if prod(ge_compared).load(0) == 1:
            return True
        return False

    fn __ge__(self, other: SIMD[dtype, 1]) raises -> Bool:
        var other_array = full(self.shape()[self.ndim() - 1], other)
        return self.__ge__(other_array)

    fn __gt__(self, other: Array) raises -> Bool:
        var gt_compared = greater(self, other)
        if prod(gt_compared).load(0) == 1:
            return True
        return False

    fn __gt__(self, other: SIMD[dtype, 1]) raises -> Bool:
        var other_array = full(self.shape()[self.ndim() - 1], other)
        return self.__gt__(other_array)

    fn __le__(self, other: Array) raises -> Bool:
        var le_compared = less_equal(self, other)
        if prod(le_compared).load(0) == 1:
            return True
        return False

    fn __le__(self, other: SIMD[dtype, 1]) raises -> Bool:
        var other_array = full(self.shape()[self.ndim() - 1], other)
        return self.__le__(other_array)

    fn __lt__(self, other: Array) raises -> Bool:
        var lt_compared = less(self, other)
        if prod(lt_compared).load(0) == 1:
            return True
        return False

    fn __lt__(self, other: SIMD[dtype, 1]) raises -> Bool:
        var other_array = full(self.shape()[self.ndim() - 1], other)
        return self.__lt__(other_array)


alias Tensor = Array
alias ndarray = Array
