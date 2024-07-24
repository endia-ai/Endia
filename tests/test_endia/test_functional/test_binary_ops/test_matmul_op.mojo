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


def run_test_matmul(msg: String = "matmul"):
    torch = Python.import_module("torch")
    arg0 = nd.randn(List(2, 30, 40))
    arg1 = nd.randn(List(40, 50))
    arg0_torch = nd.utils.to_torch(arg0)
    arg1_torch = nd.utils.to_torch(arg1)

    res = nd.matmul(arg0, arg1)
    res_torch = torch.matmul(arg0_torch, arg1_torch)

    if not nd.utils.is_close(res, res_torch):
        print("\033[31mTest failed\033[0m", msg)
    else:
        print("\033[32mTest passed\033[0m", msg)


def run_test_matmul_grad(msg: String = "matmul_grad"):
    torch = Python.import_module("torch")
    arg0 = nd.randn(List(2, 30, 40), requires_grad=True)
    arg1 = nd.randn(List(40, 50), requires_grad=True)
    arg0_torch = nd.utils.to_torch(arg0)
    arg1_torch = nd.utils.to_torch(arg1)

    res = nd.sum(nd.matmul(arg0, arg1))
    res_torch = torch.sum(torch.matmul(arg0_torch, arg1_torch))

    res.backward()
    res_torch.backward()

    grad0 = arg0.grad()
    grad1 = arg1.grad()
    grad0_torch = arg0_torch.grad
    grad1_torch = arg1_torch.grad

    if not nd.utils.is_close(grad0, grad0_torch):
        print("\033[31mTest failed\033[0m", msg, "grad0")
    if not nd.utils.is_close(grad1, grad1_torch):
        print("\033[31mTest failed\033[0m", msg, "grad1")
    if nd.utils.is_close(grad0, grad0_torch) and nd.utils.is_close(
        grad1, grad1_torch
    ):
        print("\033[32mTest passed\033[0m", msg)


def run_test_matmul_complex(msg: String = "matmul_complex"):
    torch = Python.import_module("torch")
    arg0 = nd.randn_complex(List(2, 30, 40))
    arg1 = nd.randn_complex(List(40, 50))
    arg0_torch = nd.utils.to_torch(arg0)
    arg1_torch = nd.utils.to_torch(arg1)

    res = nd.matmul(arg0, arg1)
    res_torch = torch.matmul(arg0_torch, arg1_torch)

    if not nd.utils.is_close(res, res_torch):
        print("\033[31mTest failed\033[0m", msg)
    else:
        print("\033[32mTest passed\033[0m", msg)
