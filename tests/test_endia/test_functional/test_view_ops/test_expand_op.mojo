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


def run_test_expand(msg: String = "expand"):
    torch = Python.import_module("torch")
    arr = nd.randn(List(2, 1, 40))
    arr_torch = nd.utils.to_torch(arr)

    shape = List(2, 2, 30, 40)
    shape_torch = [2, 2, 30, 40]

    res = nd.expand(arr, nd.ArrayShape(shape))
    res_torch = torch.broadcast_to(arr_torch, shape_torch)

    if not nd.utils.is_close(res, res_torch):
        print("\033[31mTest failed\033[0m", msg)
    else:
        print("\033[32mTest passed\033[0m", msg)


def run_test_expand_grad(msg: String = "expand_grad"):
    torch = Python.import_module("torch")
    arr = nd.arange(0, 2 * 30 * 1, requires_grad=True).reshape(List(2, 30, 1))
    arr_torch = nd.utils.to_torch(arr)

    res = nd.sum(nd.sin(nd.expand(arr, nd.ArrayShape(List(2, 30, 40)))))
    res_torch = torch.sum(torch.sin(arr_torch.broadcast_to((2, 30, 40))))

    res.backward()
    res_torch.backward()

    grad = arr.grad()
    grad_torch = arr_torch.grad

    if not nd.utils.is_close(grad, grad_torch):
        print("\033[31mTest failed\033[0m", msg)
    else:
        print("\033[32mTest passed\033[0m", msg)
