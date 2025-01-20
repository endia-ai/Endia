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
    copy_shape
)
from endia.utils.aliases import dtype, nelts, NA
from algorithm import vectorize, parallelize
import math

from endia.functional._utils import (
    op_array,
    setup_shape_and_data,
)
from ._utils import DifferentiableViewOp
from endia.functional import reduce_add

####--------------------------------------------------------------------------------------------------------------------####
# flip/Broadcast
####--------------------------------------------------------------------------------------------------------------------####

struct Flip():

    @staticmethod
    fn __call__(inout curr: Array, args: List[Array]) raises:
        var dims = array_shape_to_list(curr.array_shape().args()[1])

        if len(dims) == 0:
            return
        
        setup_shape_and_data(curr)
        
        var arg = args[0]
        var shape = curr.shape()
        var rank = arg.ndim()

        # convert negative dims into positive ones
        for i in range(len(dims)):
            var dim = dims[i]
            if -dim > rank:
                raise "Error: Cannot flip, because dim exceeds the rank of the input array"
            if dim < 0:
                dims[i] = rank + dim

        # execute
        if rank == 1:
            for i in range(shape[0]):
                curr.store(i, arg.load(shape[0] - i - 1))

        elif rank == 2:
            for i in range(shape[0]):
                var i_idx = shape[0] - i - 1 if 0 in dims else i

                for j in range(shape[1]):
                    var j_idx = shape[1] - j - 1 if 1 in dims else j
                    curr.store(i * shape[1] + j, arg.load(i_idx * shape[1] + j_idx))

        elif rank == 3:
            for i in range(shape[0]):
                var i_idx = shape[0] - i - 1 if 0 in dims else i

                for j in range(shape[1]):
                    var j_idx = shape[1] - j - 1 if 1 in dims else j

                    for k in range(shape[2]):
                        var k_idx = shape[2] - k - 1 if 2 in dims else k
                        curr.store(i * shape[1] * shape[2] + j * shape[2] + k, arg.load(i_idx * shape[1] * shape[2] + j_idx * shape[2] + k_idx))

    @staticmethod
    fn jvp(primals: List[Array], tangents: List[Array]) raises -> Array:
        return default_jvp(primals, tangents)

    @staticmethod
    fn vjp(primals: List[Array], grad: Array, out: Array) raises -> List[Array]:
        var dims = array_shape_to_list(out.array_shape().args()[1])
        return List(flip(grad, dims))

    @staticmethod
    fn fwd(
        arg0: Array,
        dims: List[Int],
    ) raises -> Array:
        var arr_shape = setup_array_shape(
            List(arg0.array_shape(), list_to_array_shape(dims)),
            "copy_shape",
            copy_shape,
        )

        var curr = op_array(
            arr_shape,
            List(arg0),
            NA,
            "flip",
            Flip.__call__,
            Flip.jvp,
            Flip.vjp,
            False,
        )

        return curr


fn flip(
    arg0: Array, dims: List[Int] = List(-1)
) raises -> Array:
    return Flip.fwd(arg0, dims)
