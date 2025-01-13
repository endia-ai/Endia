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

from endia import Array
from compile import *

###############################################################################################################
#                                               JIT Compilation
#
# Any Function can be traced, even those that branch via conditional statements on the data of the arrays.
# It works as follows:
#    1) Any Operation has a name (e.g. add, sub, incr, etc.). When a function is traced/captured, the
#       names of the ops are compared to the operation at the curr_idx of the execution, which points to
#       the operation that is expected, based on what has happened in an earlier capture of the function.
#       If both names match, we return immediately and postpone our execution, if not equal, we brach and
#       mark this potential different path at the curr_idx. The operations of the unexpected/new trace is
#       are added at the end of the trace of the graph. That way, over time, the Graph will kperf_counter about all
#       possible branches, while still storing/caching previous traces.
#    2) So far we have a graph with different branches. However the purpose of the graph capturing is to
#       optimize subgraphs of the computation, i.e. cache optimized subgraphs and call then when they are
#       needed/the values of their out nodes are used in the outer function.
#       General Idea: The Graph consists of GraphNodes, which is a small datastructure which is kind of a
#       dual node to the actual node in the outer function, except it does not follow the deletion of its
#       primal and lives as long as the grpah lives. If a node in the outer function is marked as a breakpoint
#       via a getter or setter, this breakpoint marks an end to a subgraph in the Graph. Everytime we come
#       across a breakpoint node, we look at the current subgraph which emerged thorugh the last
#       couple of operations and try to fuse elemtwise ops together and perform general optimizations on this
#       perf_counter static and cached computation graph. If we come across this breakpoint node again in another
#       execution of the outer function, we simply call this optimized subgraph and we are done.
###############################################################################################################


fn jit(arg: Callable, compile_with_MAX: Bool = True) raises -> Callable:
    """
    Jit and cache the given function or Callable.
    """
    return Callable(
        arg.func,
        arg.argnums,
        arg.order_of_differentiation,
        True,
        arg.keep_intermediate_outs,
        compile_with_MAX,
    )


fn jit(
    arg: fn (List[Array]) raises -> Array, compile_with_MAX: Bool = True
) raises -> Callable:
    """
    Jit and cache the given function or Callable.
    """
    return Callable(
        arg,
        List[Int](-1),
        0,
        True,
        False,
        compile_with_MAX,
    )


fn jit(
    arg: fn (Array) raises -> Array, compile_with_MAX: Bool = True
) raises -> Callable:
    """
    Jit and cache the given function or Callable.
    """
    return Callable(
        arg,
        List[Int](-1),
        0,
        True,
        False,
        compile_with_MAX,
    )
