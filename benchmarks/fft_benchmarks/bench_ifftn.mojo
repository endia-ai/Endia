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


def ifftn_benchmark():
    print("\nIFFTN Benchmark ###########################################")
    var torch = Python.import_module("torch")

    for n in range(5, 16):
        var depth = 4  # 2 ** (n - 5)
        var width = 16  # 2 ** (n - 5)
        var height = 2**n
        var size = depth * width * height

        print("Depth:", depth, " - Width:", width, " - Height:", height)

        var x = nd.complex(
            nd.arange(0, size).reshape(List(depth, width, height)),
            nd.arange(0, size).reshape(List(depth, width, height)),
        )
        var x_torch = torch.complex(
            torch.arange(0, size).float().reshape(depth, width, height),
            torch.arange(0, size).float().reshape(depth, width, height),
        )

        num_iterations = 20
        warmup = 5
        total = Float32(0)
        total_torch = Float32(0)

        for iteration in range(num_iterations + warmup):
            if iteration < warmup:
                total = 0
                total_torch = 0

            start = perf_counter()
            _ = nd.ifftn(x)
            total += perf_counter() - start

            start = perf_counter()
            _ = torch.fft.ifftn(x_torch)
            total_torch += perf_counter() - start

        my_time = total / (1000000000 * num_iterations)
        torch_time = total_torch / (1000000000 * num_iterations)
        print("Time taken:", my_time)
        print("Time taken Torch:", torch_time)
        print("Difference:", (torch_time - my_time) / torch_time * 100, "%")
        print()
