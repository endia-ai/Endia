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


def ifft_benchmark():
    print("\nIFFT Benchmark ###########################################")
    var torch = Python.import_module("torch")

    for n in range(4, 23):
        size = 2**n
        print("Size: 2**", end="")
        print(n, "=", size)
        x = nd.complex(
            nd.unsqueeze(nd.arange(0, size), List(0)),
            nd.unsqueeze(nd.arange(0, size), List(0)),
        )
        x_torch = torch.complex(
            torch.arange(0, size).float(), torch.arange(0, size).float()
        )

        num_iterations = 20
        warmup = 5
        total = Float32(0)
        total_torch = Float32(0)

        for iteration in range(num_iterations + warmup):
            if iteration < warmup:
                total = 0
                total_torch = 0

            start = now()
            _ = nd.ifft(x)
            total += now() - start

            start = now()
            _ = torch.fft.ifft(x_torch)
            total_torch += now() - start

        my_time = total / (1000000000 * num_iterations)
        torch_time = total_torch / (1000000000 * num_iterations)
        print("Time taken:", my_time)
        print("Time taken Torch:", torch_time)
        print("Difference:", (torch_time - my_time) / torch_time * 100, "%")
        print()
