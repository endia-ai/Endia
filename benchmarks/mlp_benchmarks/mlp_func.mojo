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
import endia.nn as nn
import endia.optim as optim
from endia.utils import dtype
import math
from time import now


def fill_sin_(inout curr: nd.Array, arg: nd.Array):
    """
    Inplace fill the current array with normalized sin values.
    """
    for i in range(arg.size()):
        curr.store(i, math.sin(50 * (arg.load(i) + 1) / 2))


def setup_params(
    x: nd.Array, y: nd.Array, hidden_dims: List[Int]
) -> List[nd.Array]:
    """
    Setup the parameters for the MLP model as a list of arrays.
    """
    params = List[nd.Array]()
    params.append(x)
    params.append(y)
    num_layers = len(hidden_dims) - 1
    for i in range(num_layers):
        weight = nd.rand_he_normal(
            List(hidden_dims[i], hidden_dims[i + 1]),
            fan_in=hidden_dims[i],
        )
        bias = nd.rand_he_normal(
            List(hidden_dims[i + 1]),
            fan_in=hidden_dims[i],
        )
        params.append(weight)
        params.append(bias)
    return params


def benchmark_mlp_func():
    print("\nRunning MLP benchmark in a functional eager mode with grad:")

    # define the forward function
    def fwd(args: List[nd.Array]) -> nd.Array:
        target = args[1]
        pred = nn.mlp(args)
        loss = nd.mse(pred, target)
        return loss

    # define the training loop
    batch_size = 128
    lr = 0.001
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    num_iters = 2000
    every = 500
    avg_loss = SIMD[dtype, 1](0)

    # setup input, target, params and velocity
    x = nd.Array(List(batch_size, 1))
    y = nd.Array(List(batch_size, 1))
    hidden_dims = List(1, 32, 64, 128, 128, 128, 64, 32, 1)
    args = setup_params(x, y, hidden_dims)
    m = List[nd.Array]()
    v = List[nd.Array]()
    for i in range(len(args)):
        m.append(nd.zeros_like(args[i]))
        v.append(nd.zeros_like(args[i]))

    # setup fwd and grad function as one call
    value_and_grad_fwd = nd.value_and_grad(fwd)

    # setup time variables
    start = Float64(0)
    end = Float64(0)
    time_all = Float64(0)
    fwd_start = Float64(0)
    fwd_end = Float64(0)
    time_fwd = Float64(0)
    grad_start = Float64(0)
    grad_end = Float64(0)
    time_grad = Float64(0)
    optim_start = Float64(0)
    optim_end = Float64(0)
    time_optim = Float64(0)

    # training loop
    for t in range(1, num_iters + 1):
        start = now()

        # fill input and target inplace
        nd.randu_(args[0])
        fill_sin_(args[1], args[0])

        # compute loss
        fwd_start = now()
        value_and_grad = value_and_grad_fwd(args)[List[List[nd.Array]]]
        fwd_end = now()

        loss = value_and_grad[0][0]
        avg_loss += loss.load(0)
        args_grads = value_and_grad[1]

        # update weights and biases inplace
        optim_start = now()
        for i in range(2, len(args_grads)):
            # implement adam with above variables as in the step function above
            m[i] = beta1 * m[i] + (1 - beta1) * args_grads[i]
            v[i] = beta2 * v[i] + (1 - beta2) * args_grads[i] * args_grads[i]
            m_hat = m[i] / (1 - beta1**t)
            v_hat = v[i] / (1 - beta2**t)
            args[i] -= lr * m_hat / (nd.sqrt(v_hat) + eps)

        optim_end = now()
        end = now()

        time_fwd += (fwd_end - fwd_start) / 1000000000
        time_optim += (optim_end - optim_start) / 1000000000
        time_all += (end - start) / 1000000000

        # print loss
        if t % every == 0:
            print("- Iter: ", t, " Loss: ", avg_loss / every)
            avg_loss = 0
            print(
                "  Total: ",
                time_all / every,
                " Value_and_Grad: ",
                time_fwd / every,
                " Optim: ",
                time_optim / every,
            )
            time_all = 0
            time_fwd = 0
            time_optim = 0
