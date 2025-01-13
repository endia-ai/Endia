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

from .custom_ops_examples import *
from .endia_vs_torch_vs_jax import *
from .simple_examples import *
from .viz_examples import *
from python import Python


def run_examples():
    # check if Graphviz is installed
    try:
        _ = Python.import_module("graphviz")

    except:
        raise "\033[91m[ERROR]\033[0m Graphviz not found. Please install graphviz to run the examples via: \n\033[92m magic add --pypi \"graphviz\" \033[0m"

    # run examples
    example1()
    example2()
    viz_example1()
