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

from math import isnan, isinf
from endia import Array
from endia.functional import *
from python import PythonObject, Python


@always_inline
fn to_torch(arg: Array) raises -> PythonObject:
    """
    Converts an endia Array to a torch tensor.
    """
    if arg.is_complex():
        var real = real(arg)
        var imag = imag(arg)
        var torch = Python.import_module("torch")
        var torch_real = to_torch_tensor(real).requires_grad_(
            arg.requires_grad()
        )
        var torch_imag = to_torch_tensor(imag).requires_grad_(
            arg.requires_grad()
        )
        var res = torch.complex(torch_real, torch_imag)
        return res
    return to_torch_tensor(arg)


@always_inline
fn to_torch_tensor(arg: Array) raises -> PythonObject:
    var torch = Python.import_module("torch")
    var shape = arg.shape()
    var size = 1
    for i in range(shape.size):
        size *= shape[i]
    var torch_shape = PythonObject([])
    for i in range(arg.ndim()):
        torch_shape.append(shape[i])
    var res = torch.zeros(size=torch_shape).to(torch.float64)

    var flattened = res.flatten()
    for i in range(size):
        flattened[i] = arg.load(i)

    if arg.requires_grad():
        res.requires_grad = True

    return res


@always_inline
fn is_close(
    x: Array, x_torch: PythonObject, rtol: Float32 = 10e-5
) raises -> Bool:
    """
    Checks if the values in the endia Array and the torch tensor are equal up to a relative tolerance.
    """
    var y = contiguous(x.reshape(x.size()))

    if x.is_complex():
        var real_torch = x_torch.real.reshape(x.size())
        var imag_torch = x_torch.imag.reshape(x.size())
        var data = y.data()
        var diff = Float32(0)
        var epsilon = Float32(10e-10)
        for i in range(x.size()):
            var real = data.load(2 * i)
            var imag = data.load(2 * i + 1)
            # var real_torch_val = real_torch[i].to_float64().cast[
            #     DType.float32
            # ]()
            # var imag_torch_val = imag_torch[i].to_float64().cast[
            #     DType.float32
            # ]()
            var real_torch_val = float(real_torch[i]).cast[DType.float32]()
            var imag_torch_val = float(imag_torch[i]).cast[DType.float32]()
            var magnitude = max(
                math.sqrt(real_torch_val**2 + imag_torch_val**2), epsilon
            )
            diff += (
                abs(real - real_torch_val) + abs(imag - imag_torch_val)
            ) / magnitude

        _ = y
        _ = real_torch
        _ = imag_torch

        diff /= x.size()
        if diff > rtol:
            print(
                "\033[33mWarning:\033[0m Relative difference:",
                diff,
                " given rtol:",
                rtol,
            )
            return False

        return True
    else:
        var data = y.data()
        var diff = Float32(0)
        var epsilon = Float32(1e-6)
        var real_torch = x_torch.flatten()

        for i in range(x.size()):
            var real = data.load(i)
            # var real_torch_val = real_torch[i].to_float64().cast[
            #     DType.float32
            # ]()
            var real_torch_val = float(real_torch[i]).cast[DType.float32]()
            var magnitude = max(abs(real_torch_val), epsilon)
            diff += abs(real - real_torch_val) / magnitude

        _ = y
        _ = real_torch

        diff /= x.size()
        if diff > rtol:
            print(
                "\033[33mWarning:\033[0m Relative difference:",
                diff,
                " given rtol:",
                rtol,
            )
            return False

        return True
