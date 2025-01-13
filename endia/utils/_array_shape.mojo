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

from memory import ArcPointer
from algorithm import vectorize, parallelize

# from utils import Variant
from time import perf_counter
from random import seed, random_ui64
import math
from python import Python, PythonObject


fn compute_stride(shape: List[Int]) -> List[Int]:
    var stride = List[Int]()
    for _ in range(len(shape)):
        stride.append(1)
    for i in range(len(shape) - 2, -1, -1):
        stride[i] = stride[i + 1] * shape[i + 1]
    return stride


###############################################################################################################
#                                                      Array Shape
###############################################################################################################


fn default_shape_op_fwd(
    inout curr: ArrayShape, args: List[ArrayShape]
) raises -> None:
    # print("default_shape_op_fwd")
    pass


@value
struct ShapeNode(CollectionElement):
    """
    ShapeNode is a reference-counted object designed to encapsulate the shape information of an array. It stores
    crucial details such as the list of dimensions (shape), ge_zero sizes along each dimension (stride), the starting
    index (storage_offset), and other shape-related metadata. ShapeNode plays a crucial role in facilitating efficient shape
    computations, particularly for view operations where the shape needs to be computed without accessing the actual data.
    """

    var shape: List[Int]
    var stride: List[Int]
    var storage_offset: Int
    var ndim: Int
    var size: Int
    var args: List[ArcPointer[Self]]
    var shape_op_fwd: fn (inout ArrayShape, List[ArrayShape]) raises -> None
    var is_computed: Bool
    var name: String

    fn __init__(
        inout self, shape: List[Int], stride: List[Int], storage_offset: Int
    ):
        self.shape = shape
        self.stride = stride
        self.storage_offset = storage_offset
        self.ndim = len(shape)
        self.size = 1
        for i in range(len(shape)):
            self.size *= shape[i]
        self.args = List[ArcPointer[Self]]()
        self.shape_op_fwd = default_shape_op_fwd
        self.is_computed = False
        self.name = ""
        # print("creating ShapeNode")

    # fn __del__(owned self):
    #     print("deleting ShapeNode")


@value
struct ArrayShape(CollectionElement, Stringable, EqualityComparable):
    """
    ArrayShape is a lightweight handle that provides an efficient way to work with and manage array shapes. It serves
    as a convenient wrapper around a ShapeNode instance, allowing for inexpensive copying of shapes without duplicating
    the underlying shape data. ArrayShape offers initialization methods to create instances from shape lists and stride.
    """

    var shape_node: ArcPointer[ShapeNode]

    fn __init__(
        inout self,
        shape: List[Int],
        stride: List[Int] = List[Int](),
        storage_offset: Int = 0,
    ):
        var _stride = stride
        if len(stride) != len(shape):
            _stride = List[Int]()
            for _ in range(len(shape)):
                _stride.append(1)
            for i in range(len(shape) - 2, -1, -1):
                _stride[i] = shape[i + 1] * _stride[i + 1]
        self.shape_node = ArcPointer[ShapeNode](
            ShapeNode(
                shape,
                _stride,
                storage_offset,
            )
        )

    fn __init__(inout self, shape_node: ArcPointer[ShapeNode]):
        self.shape_node = shape_node

    fn __copyinit__(inout self, other: ArrayShape):
        self.shape_node = other.shape_node

    fn __moveinit__(inout self, owned other: ArrayShape):
        self.shape_node = other.shape_node^

    fn __str__(self) -> String:
        var storage_offset: String = ""

        fn list_to_string(list: List[Int]) -> String:
            var out: String = ""
            out += "["
            for i in range(len(list)):
                out += str(list[i])
                if i < len(list) - 1:
                    out += ", "
            out += "]"
            return out

        var out: String = ""
        out += (
            storage_offset
            + "shape: "
            + list_to_string(self.shape_node[].shape)
            + "\n"
        )
        out += (
            storage_offset
            + "stride: "
            + list_to_string(self.shape_node[].stride)
            + "\n"
        )
        out += (
            storage_offset
            + "storage_offset: "
            + str(self.shape_node[].storage_offset)
            + "\n"
        )
        return out

    fn fwd(
        self,
    ) -> fn (inout ArrayShape, List[ArrayShape]) raises -> None:
        return self.shape_node[].shape_op_fwd

    fn set_fwd(
        inout self,
        fwd: fn (inout ArrayShape, List[ArrayShape]) raises -> None,
    ):
        self.shape_node[].shape_op_fwd = fwd

    fn execute_fwd(inout self, args: List[ArrayShape]) raises:
        self.shape_node[].shape_op_fwd(self, args)

    fn args_(inout self, args: List[ArrayShape]):
        self.shape_node[].args.clear()
        for arg in args:
            self.shape_node[].args.append(arg[].shape_node)

    fn args(self) -> List[ArrayShape]:
        var res = List[ArrayShape]()
        for arg in self.shape_node[].args:
            res.append(ArrayShape(arg[]))
        return res

    fn set_shape(inout self, shape: List[Int]):
        self.shape_node[].shape = shape

    fn shape(self) -> List[Int]:
        return self.shape_node[].shape

    fn set_stride(inout self, stride: List[Int]):
        self.shape_node[].stride = stride

    fn stride(self) -> List[Int]:
        return self.shape_node[].stride

    fn set_storage_offset(inout self, storage_offset: Int):
        self.shape_node[].storage_offset = storage_offset

    fn storage_offset(self) -> Int:
        return self.shape_node[].storage_offset

    fn set_ndim(inout self, ndim: Int):
        self.shape_node[].ndim = ndim

    fn ndim(self) -> Int:
        return self.shape_node[].ndim

    fn set_size(inout self, size: Int):
        self.shape_node[].size = size

    fn size(self) -> Int:
        return self.shape_node[].size

    # fn kwargs_(inout self, kwargs: List[Int]):
    #     self.shape_node[].kwargs = kwargs

    # fn kwargs(self) -> List[Int]:
    #     return self.shape_node[].kwargs

    fn is_computed(self) -> Bool:
        return self.shape_node[].is_computed

    fn is_computed_(inout self, is_computed: Bool):
        self.shape_node[].is_computed = is_computed

    fn setup(
        inout self,
        shape: List[Int],
        stride: List[Int] = List[Int](),
        storage_offset: Int = 0,
    ) raises:
        var _stride = compute_stride(shape) if len(stride) == 0 else stride
        var size = 1
        for i in range(len(shape)):
            size *= shape[i]
        self.set_shape(shape)
        self.set_stride(_stride)
        self.set_storage_offset(storage_offset)
        self.set_ndim(len(shape))
        self.set_size(size)

    fn __eq__(self, other: ArrayShape) -> Bool:
        var equal = True
        for i in range(len(self.shape())):
            if self.shape()[i] != other.shape()[i]:
                equal = False
                break
        return equal

    fn __ne__(self, other: ArrayShape) -> Bool:
        return not self.__eq__(other)


# recursive shape computation until all parents are computed
fn compute_shape(inout curr: ArrayShape, store_args: Bool = False) raises:
    """
    Recursively computes the shape of an ArrayShape.

    Args:
        curr: The ArrayShape to compute the shape of.
        store_args: Whether to store the arguments of the ArrayShape after computation, i.e. retaining the computation graph.

    Constraints:
    - The ArrayShape must have a forward function set.
    """
    if curr.ndim() == -1:
        raise "Error: Placeholder Error in compute_shape."
    if curr.is_computed() or len(curr.args()) == 0:
        return
    # print("compute shape")
    for arg in curr.args():
        var arg_shape = arg[]
        compute_shape(arg_shape, store_args)
    var fwd = curr.fwd()
    fwd(curr, curr.args())
    curr.is_computed_(True)
    if not store_args:
        curr.shape_node[].args.clear()


fn setup_array_shape(
    args: List[ArrayShape],
    name: String,
    fwd: fn (inout ArrayShape, List[ArrayShape]) raises -> None,
) raises -> ArrayShape:
    """
    Sets up an ArrayShape with the given arguments, name, and forward function. Does not compute the actual shape.

    Args:
        args: The arguments of the ArrayShape.
        name: The name of the ArrayShape.
        fwd: The forward function of the ArrayShape.

    Returns:
        The ArrayShape, with its actualy shape not computed yet.
    """
    var res_arr = ArrayShape(0)
    res_arr.args_(args)
    res_arr.set_fwd(fwd)
    res_arr.shape_node[].name = name
    return res_arr


fn array_shape_to_list(arg: ArrayShape) -> List[Int]:
    """
    Converts an ArrayShape to a list of Ints. Does not retain the stride and storage offset information.
    """
    return arg.shape()


fn list_to_array_shape(arg: List[Int]) -> ArrayShape:
    """
    Converts a list of Ints to an ArrayShape object.
    """
    return ArrayShape(arg)


fn array_shape_to_slices(arg: ArrayShape) raises -> List[Slice]:
    """
    Converts an ArrayShape to a list of slices.

    Args:
        arg: The ArrayShape to convert.

    Returns:
        A list of slices.
    """
    var data = arg.shape()
    var slices = List[Slice]()
    if len(data) % 3 != 0:
        raise "Invalid data for slices"
    for i in range(0, len(data), 3):
        slices.append(Slice(data[i], data[i + 1], data[i + 2]))
    return slices


fn slices_to_array_shape(arg: List[Slice]) -> ArrayShape:
    """
    Converts a list of slices to an ArrayShape.

    Args:
        arg: The list of slices to convert.

    Returns:
        The ArrayShape.
    """
    var data = List[Int]()
    for i in range(len(arg)):
        var arg = arg[i]
        data.append(arg.start.value() if arg.start else 0)
        data.append(arg.end.value() if arg.end else -1)
        data.append(arg.step.value())
    return ArrayShape(data)


fn copy_shape(inout curr: ArrayShape, args: List[ArrayShape]) raises:
    """
    Setups the shape of an array to be the same as another ArrayShape.

    Args:
        curr: The ArrayShape to store the result of the computation.
        args: The ArrayShape to copy.
    """
    curr.setup(args[0].shape())


fn broadcast_shapes(inout curr: ArrayShape, args: List[ArrayShape]) raises:
    """
    Computes the shape resulting from broadcasting two arrays together.

    Args:
        curr: The ArrayShape to store the result of the computation.
        args: Lhs ArrayShape, rhs ArrayShape, axes to ignore during broadcasting.

    #### Constraints:
    - The shape of each dimension of args[0] and args[1] must be equal or one of them must be 1 (seen from right to left).
    """
    var arg0 = args[0]
    var arg1 = args[1]
    var shape0 = arg0.shape_node[].shape
    var shape1 = arg1.shape_node[].shape
    var shape = List[Int]()
    var diff = len(shape0) - len(shape1)

    if diff > 0:
        # shape0 has more dimensions
        for i in range(diff):
            shape.append(shape0[i])
        for i in range(len(shape1)):
            if shape0[i + diff] == shape1[i]:
                shape.append(shape0[i + diff])
            elif shape0[i + diff] == 1:
                shape.append(shape1[i])
            elif shape1[i] == 1:
                shape.append(shape0[i + diff])
            else:
                raise "Error: Incompatible shapes for broadcasting"
    else:
        # shape1 has more dimensions
        for i in range(-diff):
            shape.append(shape1[i])
        for i in range(len(shape0)):
            if shape1[i - diff] == shape0[i]:
                shape.append(shape1[i - diff])
            elif shape1[i - diff] == 1:
                shape.append(shape0[i])
            elif shape0[i] == 1:
                shape.append(shape1[i - diff])
            else:
                raise "Error: Incompatible shapes for broadcasting"

    curr.setup(shape)


fn clone_shape_during_runtime(
    inout curr: ArrayShape, args: List[ArrayShape]
) raises:
    """
    This functions sets upt the curr array_shape to be the same as the arg array_shape.
    """
    var arg = args[0]
    curr.setup(arg.shape(), arg.stride(), arg.storage_offset())


fn clone_shape(arg0: ArrayShape) raises -> ArrayShape:
    """
    Clones the shape of an ArrayShape during runtime.

    Args:
        arg0: The ArrayShape to clone.

    Returns:
        The cloned ArrayShape.
    """
    return setup_array_shape(
        List(arg0), "clone_shape", clone_shape_during_runtime
    )
