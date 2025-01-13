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
    array_shape_to_slices,
    slices_to_array_shape,
    compute_shape,
    setup_array_shape,
)
from endia.utils.aliases import dtype, nelts, NA
from algorithm import vectorize, parallelize
import math

from endia.functional._utils import (
    op_array,
    setup_shape_and_data,
)

# from math import max, min
from endia.functional import pad
from ._utils import DifferentiableViewOp

####--------------------------------------------------------------------------------------------------------------------####
# Slice
####--------------------------------------------------------------------------------------------------------------------####


struct ArraySlice(DifferentiableViewOp):
    @staticmethod
    fn compute_shape(inout curr: ArrayShape, args: List[ArrayShape]) raises:
        """
        Computes the shape of an array after slicing.

        Args:
            curr: The ArrayShape to store the result of the computation.
            args: The ArrayShape to slice, and the list of slices encoded in an ArrayShape via the array_shape_to_slices function.
        """
        # we have the slices as the second argument in the form of an arrayshape
        var arg = args[0]
        var slices = array_shape_to_slices(args[1])
        var sliced_shape = List[Int]()
        var sliced_stride = List[Int]()
        var storage_offset = args[0].storage_offset()

        for i in range(arg.shape_node[].ndim):
            var slice = slices[i] if i < len(slices) else Slice(
                0, arg.shape_node[].shape[i], 1
            )
            slice.start = (
                max(0, slice.start.value())
                % (arg.shape_node[].shape[i] + 1) if slice.step.value()
                > 0 else min(
                    arg.shape_node[].shape[i],
                    slice.end.value() % (arg.shape_node[].shape[i] + 1),
                )
                - 1
            )
            slice.end = (
                min(arg.shape_node[].shape[i], slice.end.value())
                % (arg.shape_node[].shape[i] + 1) if slice.step.value()
                > 0 else max(0, slice.start.value()) - 1
            )
            sliced_shape.append(
                (slice.end.value() - slice.start.value() + slice.step.value() - 1)
                // slice.step.value()
            )
            sliced_stride.append(arg.shape_node[].stride[i] * slice.step.value())
            storage_offset += slice.start.value() * arg.shape_node[].stride[i]

        curr.setup(sliced_shape, sliced_stride, storage_offset)

    @staticmethod
    fn __call__(inout curr: Array, args: List[Array]) raises:
        """
        Performs the forward pass for the slice operation. It sets the base of the argument to be the base of the current array and computes the shape of the current array via its dedicated ArraySahpe fwd fucntion.

        Args:
            curr: The current array to store the result (modified in-place).
            args: The array on which the slice view is created.

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
        Computes the vector-Jacobian product for the slice operation.

        Args:
            primals: A list containing the primal input array.
            grad: The gradient of the output with respect to some scalar function.
            out: The output of the forward pass (unused in this function).

        Returns:
            A list containing the gradient with respect to the input.

        #### Note:
        The vector-Jacobian product for slice is computed by padding the gradient with zeros along the axes that were sliced.
        """
        var slices = array_shape_to_slices(out.array_shape().args()[1])
        return List(pad(grad, primals[0].array_shape(), slices))

    @staticmethod
    fn fwd(arg0: Array, slices: List[Slice]) raises -> Array:
        """
        Slices the input array based on the given slices.

        Args:
            arg0: The input array.
            slices: The slices to apply.

        Returns:
            The sliced array.
        """
        var arr_shape = setup_array_shape(
            List(arg0.array_shape(), slices_to_array_shape(slices)),
            "slice_shape",
            ArraySlice.compute_shape,
        )

        return op_array(
            arr_shape,
            List(arg0),
            NA,
            "slice",
            ArraySlice.__call__,
            ArraySlice.jvp,
            ArraySlice.vjp,
            True,
        )


fn array_slice(arg0: Array, slices: List[Slice]) raises -> Array:
    """
    Slices the input array based on the given slices.

    Args:
        arg0: The input array.
        slices: The slices to apply.

    Returns:
        The sliced array.
    """
    return ArraySlice.fwd(arg0, slices)
