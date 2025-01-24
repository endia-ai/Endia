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

from .array import *


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