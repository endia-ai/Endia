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
from time import perf_counter


def foo(args: List[nd.Array]) -> nd.Array:
    a = args[0]
    b = args[1]
    c = args[2]
    return nd.sum(
        nd.mul(
            nd.cos(nd.sin(nd.cos(nd.cos(nd.add(nd.matmul(a, b), c))))),
            nd.matmul(a, b),
        )
    )


def benchmark_foo_grad(
    msg: String = "foo", warmup_runs: Int = 5, num_runs: Int = 10
):
    # args initialization
    requires_grad = True
    a = nd.arange(0, 300 * 400, requires_grad).reshape(List(300, 400))
    b = nd.arange(0, 2 * 400 * 500, requires_grad).reshape(List(2, 400, 500))
    c = nd.arange(0, 2 * 300 * 500, requires_grad).reshape(List(2, 300, 500))
    args = List(a, b, c)

    # warm up
    for _ in range(warmup_runs):
        res = foo(args)
        res.backward()
        res.zero_grad()

    # functional calls
    total_time_forward = 0
    total_time_backward = 0
    for _ in range(num_runs):
        start = perf_counter()
        res = foo(args)
        end = perf_counter()

        # backward pass
        start2 = perf_counter()
        res.backward()
        end2 = perf_counter()

        # zero the gradients in the computation graph
        res.zero_grad()

        total_time_forward += end - start
        total_time_backward += end2 - start2

    print(
        "\033[36mBenchmark:\033[0m",
        msg,
        "forward:",
        (total_time_forward / num_runs) / 1000000000,
    )
    print(
        "          ",
        msg,
        "backward:",
        (total_time_backward / num_runs) / 1000000000,
    )
