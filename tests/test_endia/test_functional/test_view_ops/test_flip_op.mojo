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


def run_test_flip_1d(msg: String = "flip_1d"):
    torch = Python.import_module("torch")
    arr = nd.randn(List(30))
    arr_torch = nd.utils.to_torch(arr)

    res = nd.flip(arr, List(0))
    res_torch = torch.flip(arr_torch, [0])

    if not nd.utils.is_close(res, res_torch):
        print("\033[31mTest failed\033[0m", msg)
    else:
        print("\033[32mTest passed\033[0m", msg)

def run_test_flip_1d_grad(msg: String = "flip_1d_grad"):
    torch = Python.import_module("torch")
    arr = nd.randn(List(30), requires_grad=True)
    arr_torch = nd.utils.to_torch(arr)
    arr_torch.requires_grad = True

    res = nd.sum(nd.flip(arr, List(0)))
    res_torch = torch.sum(torch.flip(arr_torch, [0]))

    res.backward()
    res_torch.backward()

    grad = arr.grad()
    grad_torch = arr_torch.grad

    if not nd.utils.is_close(grad, grad_torch):
        print("\033[31mTest failed\033[0m", msg)
    else:
        print("\033[32mTest passed\033[0m", msg)

def run_test_flip_2d(msg: String = "flip_2d"):
    torch = Python.import_module("torch")
    arr = nd.randn(List(20, 30))
    arr_torch = nd.utils.to_torch(arr)

    res = nd.flip(arr, List(0, 1))
    res_torch = torch.flip(arr_torch, [0, 1])

    if not nd.utils.is_close(res, res_torch):
        print("\033[31mTest failed\033[0m", msg)
    else:
        print("\033[32mTest passed\033[0m", msg)

def run_test_flip_2d_grad(msg: String = "flip_2d_grad"):
    torch = Python.import_module("torch")
    arr = nd.randn(List(20, 30), requires_grad=True)
    arr_torch = nd.utils.to_torch(arr)
    arr_torch.requires_grad = True

    res = nd.sum(nd.flip(arr, List(0, 1)))
    res_torch = torch.sum(torch.flip(arr_torch, [0, 1]))

    res.backward()
    res_torch.backward()

    grad = arr.grad()
    grad_torch = arr_torch.grad

    if not nd.utils.is_close(grad, grad_torch):
        print("\033[31mTest failed\033[0m", msg)
    else:
        print("\033[32mTest passed\033[0m", msg)

def run_test_flip_3d(msg: String = "flip_3d"):
    torch = Python.import_module("torch")
    arr = nd.randn(List(10, 20, 30))
    arr_torch = nd.utils.to_torch(arr)

    res = nd.flip(arr, List(0, 1, 2))
    res_torch = torch.flip(arr_torch, [0, 1, 2])

    if not nd.utils.is_close(res, res_torch):
        print("\033[31mTest failed\033[0m", msg)
    else:
        print("\033[32mTest passed\033[0m", msg)

def run_test_flip_3d_grad(msg: String = "flip_3d_grad"):
    torch = Python.import_module("torch")
    arr = nd.randn(List(10, 20, 30), requires_grad=True)
    arr_torch = nd.utils.to_torch(arr)
    arr_torch.requires_grad = True

    res = nd.sum(nd.flip(arr, List(0, 1, 2)))
    res_torch = torch.sum(torch.flip(arr_torch, [0, 1, 2]))

    res.backward()
    res_torch.backward()

    grad = arr.grad()
    grad_torch = arr_torch.grad

    if not nd.utils.is_close(grad, grad_torch):
        print("\033[31mTest failed\033[0m", msg)
    else:
        print("\033[32mTest passed\033[0m", msg)
