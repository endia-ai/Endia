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
from endia.utils import (
    array_shape_to_list,
    compute_stride,
    setup_array_shape,
    list_to_array_shape,
)
from endia.utils.aliases import dtype, nelts, NA
from algorithm import vectorize, parallelize
import math
from endia.functional._utils import contiguous

from endia.functional._utils import (
    op_array,
    setup_shape_and_data,
)
from ._utils import DifferentiableReduceOp
from endia.functional import expand
from endia.functional import squeeze
from utils.numerics import max_or_inf

####-----------------------------------------------------------------------------------------------------------------####
#### Reduce Ops
####-----------------------------------------------------------------------------------------------------------------####


struct ReduceArgMin(DifferentiableReduceOp):
    @staticmethod
    fn compute_shape(inout curr: ArrayShape, args: List[ArrayShape]) raises:
        """
        Computes the shape of an array after reducing along a specific axis.

        Args:
            curr: The ArrayShape to store the result of the computation.
            args: The ArrayShape to reduce, and the axis to reduce along encoded in an ArrayShape.

        #### Constraints:
        - The axis must be a valid axis of the ArrayShape (args[0]).
        - The number of axis must not exceed the number of dimensions of the ArrayShape (args[0]).
        """
        var arg = args[0]
        var axis = array_shape_to_list(args[1])
        var shape = arg.shape_node[].shape
        var new_shape = List[Int]()
        for i in range(len(shape)):
            if not list_contains(axis, i):
                new_shape.append(shape[i])
            else:
                new_shape.append(1)
        curr.setup(new_shape)

    @staticmethod
    fn __call__(inout curr: Array, args: List[Array]) raises:
        """
        Performs the forward pass for element-wise arg_minition of two arrays.

        Computes the sum of the input arrays and stores the result in the current array.
        Initializes the current array if not already set up.

        Args:
            curr: The current array to store the result (modified in-place).
            args: A list containing the input arrays.

        #### Note:
        This function assumes that the shape and data of the args are already set up.
        If the current array (curr) is not initialized, it computes the shape based on the input array and the axis and sets up the data accordingly.
        """
        setup_shape_and_data(curr)
        var arg = contiguous(args[0])
        var arg_shape = arg.shape()
        var arg_stride = arg.stride()
        var target_shape = curr.shape()
        var rank = curr.ndim()
        var target_stride = compute_stride(target_shape)
        var arg_min_stride = target_stride
        var denom = 1
        for i in range(rank):
            if target_shape[i] == 1 and arg_shape[i] != 1:
                target_stride[i] = 0
                denom *= arg_stride[i]
            else:
                arg_min_stride[i] = 0

        fill_(curr, -1)
        var tmp_min = full(target_shape, max_or_inf[dtype]())

        var target_storage_offset = curr.storage_offset()
        var tmp_min_data = tmp_min.data()
        var arg_data = arg.data()
        var curr_data = curr.data()

        if rank != 1:
            # check if both shapes are actually equal and we simply have to perdorm a fast copy
            var rows = arg_shape[rank - 2]
            var cols = arg_shape[rank - 1]

            for i in range(0, arg.size(), rows * cols):
                var nd_idx = compute_nd_index(i, arg_shape)
                var target_idx_0 = compute_storage_offset(
                    nd_idx, target_stride, target_storage_offset
                )
                var arg_min_idx_0 = compute_storage_offset(
                    nd_idx, arg_min_stride, target_storage_offset
                )
                for j in range(rows):
                    var base_idx_1 = i + j * arg_stride[rank - 2]
                    var target_idx_1 = target_idx_0 + j * target_stride[
                        rank - 2
                    ]
                    var arg_min_idx_1 = arg_min_idx_0 + j * arg_min_stride[
                        rank - 2
                    ]

                    for k in range(cols):
                        var base_idx = base_idx_1 + k * arg_stride[rank - 1]
                        var target_idx = target_idx_1 + k * target_stride[
                            rank - 1
                        ]
                        var arg_min_idx = arg_min_idx_1 + k * arg_min_stride[
                            rank - 1
                        ]

                        var new_curr = min(
                            tmp_min_data.load(target_idx),
                            arg_data.load(base_idx),
                        )
                        if new_curr < tmp_min_data.load(target_idx):
                            curr_data.store(target_idx, arg_min_idx // denom)

                        tmp_min_data.store(target_idx, new_curr)
        else:
            # if the rank is one and we still want to reduce along the single axis
            if target_stride[0] == 0:
                # var end = arg.size() - arg.size() % nelts[dtype]()
                for i in range(0, arg.size()):
                    var new_curr = min(tmp_min_data.load(0), arg_data.load(i))
                    if new_curr != tmp_min_data.load(0):
                        curr_data.store(0, i)
                    tmp_min_data.store(0, new_curr)
            # otherwise, if we we have rank one but not´reduction, we simply copy the values
            else:
                var end = arg.size() - arg.size() % nelts[dtype]()
                for i in range(0, end, nelts[dtype]()):
                    tmp_min_data.store[width = nelts[dtype]()](
                        i,
                        arg_data.load[width = nelts[dtype]()](i).reduce_min(),
                    )
                for i in range(end, arg.size()):
                    tmp_min_data.store(i, arg_data.load(i))

        _ = arg
        _ = tmp_min

    @staticmethod
    fn jvp(primals: List[Array], tangents: List[Array]) raises -> Array:
        return default_jvp(primals, tangents)

    @staticmethod
    fn vjp(primals: List[Array], grad: Array, out: Array) raises -> List[Array]:
        return default_vjp(primals, grad, out)

    @staticmethod
    fn fwd(arg0: Array, axis: List[Int]) raises -> Array:
        """
        Reduces the input array along the specified axis by summing the elements.

        Args:
            arg0: The input array.
            axis: The axis along which to reduce the array.

        Returns:
            An array containing the sum of the input array along the specified axis.

        #### Examples:
        ```python
        a = Array([[1, 2], [3, 4]])
        result = reduce_argmin(a, List(0))
        print(result)
        ```

        #### Note:
        This function supports:
        - Automatic differentiation (forward and reverse modes).
        - Complex valued arguments.
        """
        if arg0.is_complex():
            raise "Warning in reduce_argmin:  The reduce_argmin operation does not support complex numbers."

        var arr_shape = setup_array_shape(
            List(arg0.array_shape(), list_to_array_shape(axis)),
            "reduce_arg_min",
            ReduceArgMin.compute_shape,
        )

        return op_array(
            arr_shape,
            List(arg0),
            NA,
            "reduce_arg_min",
            ReduceArgMin.__call__,
            ReduceArgMin.jvp,
            ReduceArgMin.vjp,
        )


fn reduce_argmin(
    arg0: Array, axis: List[Int], keepdims: Bool = False
) raises -> Array:
    """
    Reduces the input array along the specified axis by summing the elements.

    Args:
        arg0: The input array.
        axis: The axis along which to reduce the array.

    Returns:
        An array containing the sum of the input array along the specified axis.

    #### Examples:
    ```python
    a = Array([[1, 2], [3, 4]])
    result = reduce_argmin(a, List(0))
    print(result)
    ```

    #### Note:
    This function supports:
    - Automatic differentiation (forward and reverse modes).
    - Complex valued arguments.
    """
    if keepdims:
        return ReduceArgMin.fwd(arg0, axis)
    return squeeze(ReduceArgMin.fwd(arg0, axis))
