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


struct MLP(StringableRaising):
    var weights: List[nd.Array]
    var biases: List[nd.Array]
    var hidden_dims: List[Int]
    var num_layers: Int
    var compute_backward: Bool

    fn __init__(
        inout self, hidden_dims: List[Int], compute_backward: Bool = False
    ) raises:
        self.weights = List[nd.Array]()
        self.biases = List[nd.Array]()
        self.hidden_dims = hidden_dims
        self.num_layers = len(hidden_dims) - 1
        self.compute_backward = compute_backward

        for i in range(self.num_layers):
            var weight = nd.rand_he_normal(
                List(hidden_dims[i], hidden_dims[i + 1]),
                fan_in=hidden_dims[i],
                requires_grad=compute_backward,
            )
            var bias = nd.rand_he_normal(
                List(hidden_dims[i + 1]),
                fan_in=hidden_dims[i],
                requires_grad=compute_backward,
            )
            self.weights.append(weight)
            self.biases.append(bias)

    fn forward(self, x: nd.Array) raises -> nd.Array:
        var pred = x
        for i in range(self.num_layers):
            pred = pred @ self.weights[i] + self.biases[i]
            if i < self.num_layers - 1:
                pred = nd.relu(pred)
        return pred

    fn params(self) raises -> List[nd.Array]:
        var params = List[nd.Array]()
        for i in range(self.num_layers):
            params.append(self.weights[i])
            params.append(self.biases[i])
        return params

    fn __str__(self) raises -> String:
        var out = str("")
        for i in range(self.num_layers):
            out += "Layer " + str(i) + "\n"
            out += self.weights[i].__str__() + "\n"
            out += self.biases[i].__str__() + "\n"
        return out


fn mlp(args: List[nd.Array]) raises -> nd.Array:
    var pred = args[0]
    var num_layers = (len(args) - 2) // 2
    for i in range(num_layers):
        var weight = args[i * 2 + 2]
        var bias = args[i * 2 + 3]
        pred = pred @ weight + bias
        if i < num_layers - 1:
            pred = nd.relu(pred)
        # print("\n\nLayer ", i,":")
        # print(str(pred))
    return pred
