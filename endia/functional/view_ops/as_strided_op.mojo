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
from endia.utils import setup_array_shape
from endia.utils.aliases import dtype, nelts, NA
from algorithm import vectorize, parallelize
import math
from endia.functional._utils import contiguous

from endia.functional._utils import (
    op_array,
    setup_shape_and_data,
)
from ._utils import DifferentiableViewOp

####--------------------------------------------------------------------------------------------------------------------####
# As AsStrided
####--------------------------------------------------------------------------------------------------------------------####


struct AsStrided(DifferentiableViewOp):
    @staticmethod
    fn compute_shape(inout curr: ArrayShape, args: List[ArrayShape]) raises:
        """
        Computes the shape of an array after striding.

        Args:
            curr: The ArrayShape to store the result of the computation.
            args: The ArrayShape to stride, the shape, stride and storage offset of the target ArrayShape encoded in a  single ArrayShape.
        """
        # like the slice method however we only take the real part of the array
        var arg = args[1]
        curr.setup(arg.shape(), arg.stride(), arg.storage_offset())

    @staticmethod
    fn __call__(inout curr: Array, args: List[Array]) raises:
        """
        Performs the forward pass for the as_strided operation. It sets the base of the argument to be the base of the current array and computes the shape of the current array via its dedicated ArraySahpe fwd fucntion.

        Args:
            curr: The current array to store the result (modified in-place).
            args: The array on which the as_strided view is created.

        #### Note:
        The information of the shape computation is stored in the ArrayShape object of the curr array.
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
        Computes the vector-Jacobian product for the as_strided operation.

        Args:
            primals: A list containing the primal input array.
            grad: The gradient of the output with respect to some scalar function.
            out: The output of the forward pass.

        Returns:
            A list containing the gradient with respect to the input.

        #### Note:
        The vector-Jacobian product for as_strided is computed by calling the inverse operation as_strided_inv.
        """
        var out_shape = out.array_shape()
        return as_strided_inv(
            grad,
            ArrayShape(primals[0].shape()),
            out_shape.shape(),
            out_shape.stride(),
            out_shape.storage_offset(),
        )

    @staticmethod
    fn fwd(
        arg0: Array,
        shape: List[Int],
        stride: List[Int],
        storage_offset: Int,
    ) raises -> Array:
        """
        Creates a view of the input array with the given shape and stride.

        Args:
            arg0: The input array.
            shape: The shape of the view.
            stride: The stride of the view.
            storage_offset: The storage offset of the view.

        Returns:
            A view of the input array with the given shape and stride.
        """
        var arr_shape = setup_array_shape(
            List(
                arg0.array_shape(),
                ArrayShape(shape, stride, storage_offset),
            ),
            "as_strided_shape",
            AsStrided.compute_shape,
        )

        return op_array(
            arr_shape,
            List(arg0),
            NA,
            "as_strided",
            AsStrided.__call__,
            AsStrided.jvp,
            AsStrided.vjp,
            True,
        )


fn as_strided(
    arg0: Array, shape: List[Int], stride: List[Int], storage_offset: Int
) raises -> Array:
    """
    Creates a view of the input array with the given shape and stride.

    Args:
        arg0: The input array.
        shape: The shape of the view.
        stride: The stride of the view.
        storage_offset: The storage offset of the view.

    Returns:
    A view of the input array with the given shape and stride.
    """
    return AsStrided.fwd(arg0, shape, stride, storage_offset)


####--------------------------------------------------------------------------------------------------------------------####
# As AsStrided Inverse
####--------------------------------------------------------------------------------------------------------------------####


struct AsStridedInv(DifferentiableViewOp):
    @staticmethod
    fn compute_shape(inout curr: ArrayShape, args: List[ArrayShape]) raises:
        """
        Computes the shape of an array after striding, in an inverse manner to the as_strided_shape function.

        Args:
            curr: The ArrayShape to store the result of the computation.
            args: The ArrayShape to stride, the shape, stride and storage offset of the target ArrayShape encoded in a  single ArrayShape.
        """
        # liek the slice method however we only take the real part of the array
        var arg = args[0]
        curr.setup(arg.shape(), arg.stride(), arg.storage_offset())

    @staticmethod
    fn __call__(inout curr: Array, args: List[Array]) raises:
        """
        Performs the forward pass for the as_strided_inv operation. It sets the base of the argument to be the base of the current array and computes the shape of the current array via its dedicated ArraySahpe fwd fucntion.

        Args:
            curr: The current array to store the result (modified in-place).
            args: The array on which the as_strided_inv view is created.

        #### Note:
        The information of the shape computation is stored in the ArrayShape object of the curr array.
        """
        var arg = contiguous(args[0])
        var target_array_shape = curr.array_shape().args()[1]
        setup_shape_and_data(curr)
        var curr_strided = as_strided(
            curr,
            shape=target_array_shape.shape(),
            stride=target_array_shape.stride(),
            storage_offset=target_array_shape.storage_offset(),
        )
        for i in range(arg.size()):
            curr_strided.store(i, curr_strided.load(i) + arg.load(i))

    @staticmethod
    fn jvp(primals: List[Array], tangents: List[Array]) raises -> Array:
        return default_jvp(primals, tangents)

    @staticmethod
    fn vjp(primals: List[Array], grad: Array, out: Array) raises -> List[Array]:
        """
        Computes the vector-Jacobian product for the as_strided_inv operation.

        Args:
            primals: A list containing the primal input array.
            grad: The gradient of the output with respect to some scalar function.
            out: The output of the forward pass.

        Returns:
            A list containing the gradient with respect to the input.

        #### Note:
        The vector-Jacobian product for as_strided_inv is computed by calling the as_strided operation.
        """
        var target_array_shape = grad.array_shape().args()[1]
        return List(
            as_strided(
                grad,
                shape=target_array_shape.shape(),
                stride=target_array_shape.stride(),
                storage_offset=target_array_shape.storage_offset(),
            )
        )

    @staticmethod
    fn fwd(
        arg0: Array,
        target_shape: ArrayShape,
        shape: List[Int],
        stride: List[Int],
        storage_offset: Int,
    ) raises -> Array:
        """
        Creates a view of the input array with the given shape and stride.

        Args:
            arg0: The input array.
            target_shape: The target shape of the view.
            shape: The shape of the view.
            stride: The stride of the view.
            storage_offset: The storage offset of the view.

        Returns:
            A view of the input array with the given shape and stride.
        """
        var arr_shape = setup_array_shape(
            List(target_shape, ArrayShape(shape, stride, storage_offset)),
            "as_strided_inv_shape",
            AsStridedInv.compute_shape,
        )

        var curr = op_array(
            arr_shape,
            List(arg0),
            NA,
            "as_strided_inv",
            AsStridedInv.__call__,
            AsStridedInv.jvp,
            AsStridedInv.vjp,
            False,
        )
        return curr


fn as_strided_inv(
    arg0: Array,
    target_shape: ArrayShape,
    shape: List[Int],
    stride: List[Int],
    storage_offset: Int,
) raises -> Array:
    """
    Creates a view of the input array with the given shape and stride.

    Args:
        arg0: The input array.
        target_shape: The target shape of the view.
        shape: The shape of the view.
        stride: The stride of the view.
        storage_offset: The storage offset of the view.

    Returns:
    A view of the input array with the given shape and stride.
    """
    return AsStridedInv.fwd(arg0, target_shape, shape, stride, storage_offset)
