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

import endia as nd
from python import Python


def run_test_reduce_min(msg: String = "reduce_min"):
    torch = Python.import_module("torch")
    arr = nd.randu(List(2, 30, 40))
    arr_torch = nd.utils.to_torch(arr)

    axis = 1  # in PyTorch we can only call min and argmin along a single dimension at a time!

    res = nd.reduce_min(arr, axis)
    res_torch = torch.min(arr_torch, dim=axis)[
        0
    ]  # PyTorch always returns both the min and argmin as a Tuple

    if not nd.utils.is_close(res, res_torch):
        print("\033[31mTest failed\033[0m", msg)
    else:
        print("\033[32mTest passed\033[0m", msg)
