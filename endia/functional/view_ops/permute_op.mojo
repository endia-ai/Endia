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
    setup_array_shape,
    array_shape_to_list,
    list_to_array_shape,
)
from endia.utils.aliases import dtype, nelts, NA
from algorithm import vectorize, parallelize
import math

from endia.functional._utils import (
    op_array,
    setup_shape_and_data,
)
from ._utils import DifferentiableViewOp

####--------------------------------------------------------------------------------------------------------------------####
# Permute/Transpose/Swapaxes/Swapdims
####--------------------------------------------------------------------------------------------------------------------####


struct Permute(DifferentiableViewOp):
    @staticmethod
    fn compute_shape(inout curr: ArrayShape, args: List[ArrayShape]) raises:
        """
        Permutes the dimensions of an array shape given a list of axes.

        Args:
            curr: The ArrayShape to store the result of the computation.
            args: The ArrayShape to permute, and the list of axes to permute.

        #### Constraints:
        - The number of axes in the list must not exceed the number of dimensions of the ArrayShape.
        """
        var arg = args[0]
        var axis = array_shape_to_list(args[1])
        var perm_axis = List[Int]()
        var ndim = arg.shape_node[].ndim
        if len(axis) > ndim:
            raise "Error: Number of axes in the list exceeds the number of dimensions of the ArrayShape"
        for i in range(ndim):
            perm_axis.append(i)
        for i in range(len(axis)):
            var a = axis[i]
            if a < 0:
                a = ndim + a
            perm_axis[(ndim - len(axis)) + i] = a
        var shape = arg.shape_node[].shape
        var stride = arg.shape_node[].stride
        var storage_offset = arg.shape_node[].storage_offset
        var new_shape = List[Int]()
        var new_stride = List[Int]()

        for i in range(len(perm_axis)):
            new_shape.append(shape[perm_axis[i]])
            new_stride.append(stride[perm_axis[i]])

        curr.setup(new_shape, new_stride, storage_offset)

    @staticmethod
    fn __call__(inout curr: Array, args: List[Array]) raises:
        """
        Permutes the input array based on the given axis and stores the result in the current array (curr). The first agument is set as the base of the current array.

        Args:
            curr: The current array, must be mutable.
            args: The input array and the axis to permute.

        Constraints:
            The axis must be a valid permutation of the input array's dimensions.
        """
        var array_shape = curr.array_shape()
        compute_shape(array_shape, curr.requires_grad() or curr.has_fxgraph())

    @staticmethod
    fn jvp(primals: List[Array], tangents: List[Array]) raises -> Array:
        return default_jvp(primals, tangents)

    @staticmethod
    fn vjp(primals: List[Array], grad: Array, out: Array) raises -> List[Array]:
        """
        Compute vector-Jacobian product for array permutation.

        Args:
            primals: Primal input arrays.
            grad: Gradient of the output with respect to some scalar function.
            out: The output of the forward pass.

        Returns:
            List[Array]: Gradients with respect to each input.

        #### Note:
        Implements reverse-mode automatic differentiation for permutation.
        Returns arrays with shape zero for inputs that do not require gradients.

        #### See Also:
        permute_jvp: Forward-mode autodiff for permutation.
        """
        var axis = out.array_shape().args()[1]
        return List(permute_inv(grad, axis))

    @staticmethod
    fn fwd(arg0: Array, axis: ArrayShape) raises -> Array:
        """
        Creates a view of the input array with its dimensions permuted based on the given axis.

        Args:
            arg0: The input array.
            axis: The axis to permute.

        Returns:
            A view of the input array with its dimensions permuted.
        """
        var arr_shape = setup_array_shape(
            List(arg0.array_shape(), axis),
            "permute",
            Permute.compute_shape,
        )
        var curr = op_array(
            arr_shape,
            List(arg0),
            NA,
            "permute",
            Permute.__call__,
            Permute.jvp,
            Permute.vjp,
            True,
        )
        curr.base_(arg0.base())
        return curr


fn permute(arg0: Array, axis: ArrayShape) raises -> Array:
    """
    Creates a view of the input array with its dimensions permuted based on the given axis.

    Args:
        arg0: The input array.
        axis: The axis to permute.

    Returns:
        A view of the input array with its dimensions permuted.

    #### Examples:
    ```python
     a = Array([[1, 2], [3, 4]])
     result = permute(a, axis=List(-1,-2))
     print(result)
    ```

    #### This function supports
    - Automatic differentiation (forward and reverse modes).
    - Complex valued arguments.
    """
    return Permute.fwd(arg0, axis)


fn transpose(arg0: Array, axis1: Int, axis2: Int) raises -> Array:
    """
    Transposes the input array based on the given axes.

    Args:
        arg0: The input array.
        axis1: The first axis to transpose.
        axis2: The second axis to transpose.

    Returns:
        The transposed array.

    #### Note:
    This function is a wrapper around the permute function with the given axes.
    """
    var ndim = arg0.ndim()
    var axis = List[Int]()
    for i in range(ndim):
        if i == axis1:
            axis.append(axis2)
        elif i == axis2:
            axis.append(axis1)
        else:
            axis.append(i)
    return permute(arg0, ArrayShape(axis))


fn swapaxes(arg0: Array, axis1: Int, axis2: Int) raises -> Array:
    """
    Swaps the input array's axes based on the given axes.

    Args:
        arg0: The input array.
        axis1: The first axis to swap.
        axis2: The second axis to swap.

    Returns:
        The array with the axes swapped.

    #### Note:
    This function is a wrapper around the transpose function with the given axes.
    """
    return transpose(arg0, axis1, axis2)


fn swapdims(arg0: Array, axis1: Int, axis2: Int) raises -> Array:
    """
    Swaps the input array's dimensions based on the given axes.

    Args:
        arg0: The input array.
        axis1: The first axis to swap.
        axis2: The second axis to swap.

    Returns:
        The array with the dimensions swapped.

    #### Note:
    This function is a wrapper around the transpose function with the given axes.
    """
    return swapaxes(arg0, axis1, axis2)


####--------------------------------------------------------------------------------------------------------------------####
# Inverse Permute
####--------------------------------------------------------------------------------------------------------------------####


struct InvPermute(DifferentiableViewOp):
    @staticmethod
    fn compute_shape(inout curr: ArrayShape, args: List[ArrayShape]) raises:
        """
        Permutes the dimensions of an array shape given a list of axes, in an inverse manner to the permute_shape function.

        Args:
            curr: The ArrayShape to store the result of the computation.
            args: The ArrayShape to permute, and the list of axes to permute.

        #### Constraints:
        - The number of axes in the list must not exceed the number of dimensions of the ArrayShape.
        """
        var arg = args[0]
        var axis = array_shape_to_list(args[1])
        var perm_axis = List[Int]()
        var ndim = arg.shape_node[].ndim
        if len(axis) > ndim:
            raise "Error: Number of axes in the list exceeds the number of dimensions of the ArrayShape"
        for i in range(ndim):
            perm_axis.append(i)
        for i in range(len(axis)):
            var a = axis[i]
            if a < 0:
                a = ndim + a
            perm_axis[a] = (ndim - len(axis)) + i
        var shape = arg.shape_node[].shape
        var stride = arg.shape_node[].stride
        var storage_offset = arg.shape_node[].storage_offset
        var new_shape = List[Int]()
        var new_stride = List[Int]()

        for i in range(len(perm_axis)):
            new_shape.append(shape[perm_axis[i]])
            new_stride.append(stride[perm_axis[i]])

        curr.setup(new_shape, new_stride, storage_offset)

    @staticmethod
    fn __call__(inout curr: Array, args: List[Array]) raises:
        """
        Permutes the input array based on the given axis and stores the result in the current array (curr). The first agument is set as the base of the current array.

        Args:
            curr: The current array, must be mutable.
            args: The input array and the axis to permute.

        Constraints:
            The axis must be a valid permutation of the input array's dimensions.
        """
        curr.base_(args[0].base())
        var array_shape = curr.array_shape()
        compute_shape(array_shape, curr.requires_grad() or curr.has_fxgraph())

    @staticmethod
    fn jvp(primals: List[Array], tangents: List[Array]) raises -> Array:
        return default_jvp(primals, tangents)

    @staticmethod
    fn vjp(primals: List[Array], grad: Array, out: Array) raises -> List[Array]:
        """
        Compute vector-Jacobian product for array permutation.

        Args:
            primals: Primal input arrays.
            grad: Gradient of the output with respect to some scalar function.
            out: The output of the forward pass.

        Returns:
            List[Array]: Gradients with respect to each input.

        #### Note:
        Implements reverse-mode automatic differentiation for permutation.
        Returns arrays with shape zero for inputs that do not require gradients.

        #### See Also:
        permute_inv_jvp: Forward-mode autodiff for permutation.
        """
        var axis = out.array_shape().args()[1]
        return List(permute(grad, axis))

    @staticmethod
    fn fwd(arg0: Array, axis: ArrayShape) raises -> Array:
        """
        Creates a view of the input array with its dimensions permuted based on the given axis.

        Args:
            arg0: The input array.
            axis: The axis to permute.

        Returns:
            A view of the input array with its dimensions permuted.

        #### Examples:
        ```python
        a = Array([[1, 2], [3, 4]])
        result = permute_inv(a, axis=List(-1,-2))
        print(result)
        ```

        #### This function supports
        - Automatic differentiation (forward and reverse modes).
        - Complex valued arguments.
        """
        var arr_shape = setup_array_shape(
            List(arg0.array_shape(), axis),
            "permute_inv",
            InvPermute.compute_shape,
        )
        return op_array(
            arr_shape,
            List(arg0),
            NA,
            "permute_inv",
            InvPermute.__call__,
            InvPermute.jvp,
            InvPermute.vjp,
            True,
        )


fn permute_inv(arg0: Array, axis: ArrayShape) raises -> Array:
    """
    Creates a view of the input array with its dimensions permuted based on the given axis.

    Args:
        arg0: The input array.
        axis: The axis to permute.

    Returns:
        A view of the input array with its dimensions permuted.

    #### Examples:
    ```python
    a = Array([[1, 2], [3, 4]])
    result = permute_inv(a, axis=List(-1,-2))
    print(result)
    ```

    #### This function supports
    - Automatic differentiation (forward and reverse modes).
    - Complex valued arguments.
    """
    return InvPermute.fwd(arg0, axis)
