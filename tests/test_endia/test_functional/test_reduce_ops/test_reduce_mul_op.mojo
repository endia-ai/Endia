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


def run_test_reduce_mul(msg: String = "reduce_mul"):
    torch = Python.import_module("torch")
    arr = nd.randn(List(2, 30, 40))
    arr_torch = nd.utils.to_torch(arr)

    axis = List(1)
    axis_torch = 1

    res = nd.reduce_mul(arr, axis)
    res_torch = torch.prod(arr_torch, axis_torch)

    if not nd.utils.is_close(res, res_torch):
        print("\033[31mTest failed\033[0m", msg)
    else:
        print("\033[32mTest passed\033[0m", msg)


#  TODO: Implement teh reduce_mul vjp and jvp
# def run_test_reduce_mul_grad(msg: String = "reduce_mul_grad"):
#     torch = Python.import_module("torch")
#     arr = nd.randn(List(2, 30, 40), requires_grad=True)
#     arr_torch = nd.utils.to_torch(arr)

#     axis = List(1)
#     axis_torch = [1]

#     res = nd.prod(nd.reduce_mul(arr, axis))
#     res_torch = torch.prod(torch.prod(arr_torch, axis_torch))

#     res.backward()
#     res_torch.backward()

#     grad = arr.grad()
#     grad_torch = arr_torch.grad

#     if not nd.utils.is_close(grad, grad_torch):
#         print("\033[31mTest failed\033[0m", msg)
#     else:
#         print("\033[32mTest passed\033[0m", msg)
