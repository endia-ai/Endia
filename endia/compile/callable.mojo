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
from endia.utils import list_contains
from endia.functional._utils import copy, execute_copy_raw
from endia.utils import dtype, remove_grad_rec
from utils import Variant


@value
struct Callable(CollectionElement):
    """
    Callable is the main data structure for Just-In-Time (JIT) compiling a function and computing gradients
    in a functional manner. It encapsulates the function, its arguments, and the captured computation graph,
    enabling dynamic optimization and execution.
    """

    var args: ArcPointer[List[Array]]
    var argnums: List[List[Int]]
    var func: Variant[
        fn (List[Array]) raises -> Array, fn (Array) raises -> Array
    ]
    var captured_graph: ArcPointer[FxGraph]
    var order_of_differentiation: Int
    var optimize_jit: Bool
    var args_initialized: Bool
    var keep_intermediate_outs: Bool

    fn __init__(
        inout self,
        func: Variant[
            fn (List[Array]) raises -> Array, fn (Array) raises -> Array
        ],
        argnums: List[List[Int]],
        order_of_differentiation: Int = 0,
        optimize_jit: Bool = True,
        keep_intermediate_outs: Bool = False,
        compile_with_MAX: Bool = False,
    ) raises:
        self.args = List[Array]()
        self.argnums = argnums
        self.func = func
        self.captured_graph = ArcPointer[FxGraph](FxGraph(compile_with_MAX))
        self.order_of_differentiation = order_of_differentiation
        self.optimize_jit = optimize_jit
        self.args_initialized = False
        self.keep_intermediate_outs = keep_intermediate_outs

    fn __call__(
        self, args: List[Array]
    ) raises -> Variant[Array, List[Array], List[List[Array]]]:
        var captured_graph = self.captured_graph
        if len(self.args[]) == 0:
            if self.optimize_jit:
                for _ in range(len(args)):
                    var arg = Array(List[Int](0))
                    arg.requires_grad_(True)
                    captured_graph[].op_arrayeration(arg)
                    captured_graph[].trace[-1].is_breakpoint = True
                    arg.graph_(captured_graph)
                    var cached_args = self.args
                    cached_args[].append(arg)
        else:
            if self.optimize_jit:
                if len(args) != len(self.args[]):
                    raise "Number of arguments inconcistent in jit."

        var adapted_args = List[Array]()

        if self.optimize_jit:
            captured_graph[].curr_idx = len(args)
            for i in range(len(args)):
                var arg_in = args[i]
                var arg = self.args[][i]
                var size = arg_in.base().size()
                if size != arg.size():
                    arg.data_(UnsafePointer[Scalar[dtype]].alloc(size))
                execute_copy_raw(
                    arg_in.data(),
                    arg.data(),
                    arg_in.array_shape(),
                    arg_in.is_complex(),
                )
                arg.array_shape_(arg_in.array_shape())
                captured_graph[].trace[arg.id_in_graph()].is_computed = True

            adapted_args = self.args[]

        else:
            var tmp_requires_grad_info = List[Bool]()
            for arg in args:
                tmp_requires_grad_info.append(arg[].requires_grad())
            for i in range(len(args)):
                var arg = args[i] if len(args[i].args()) == 0 else copy(args[i])
                for order in range(self.order_of_differentiation):
                    if (
                        list_contains(self.argnums[order], i)
                        or self.argnums[order][0] == -1
                    ):
                        arg.requires_grad_(True)
                adapted_args.append(arg)

        # compute forward
        var res: Array
        if self.func.isa[fn (List[Array]) raises -> Array]():
            var _func = self.func[fn (List[Array]) raises -> Array]
            res = _func(adapted_args)
        elif self.func.isa[fn (Array) raises -> Array]():
            var _func = self.func[fn (Array) raises -> Array]
            res = _func(adapted_args[0])
        else:
            raise "Function type not supported."

        # set breakpoint for the result
        # _ = res.item(0)
        res.postpone_as_grpah_output()

        var outs = List[Array]()
        outs.append(res)
        var next_outs = outs
        var number_outs_per_order = List(1)

        # compute backward from all current outs
        for order in range(self.order_of_differentiation):
            var tmp_outs = next_outs
            next_outs.clear()

            for arr in tmp_outs:
                (arr[]).backward(
                    create_graph=True if (
                        order < self.order_of_differentiation - 1
                        or self.optimize_jit
                    ) else False
                )
                for i in range(len(adapted_args)):
                    if (
                        list_contains(self.argnums[order], i)
                        or self.argnums[order][0] == -1
                    ):
                        next_outs.append(adapted_args[i].grad())

                remove_grad_rec(arr[])

            if self.keep_intermediate_outs:
                outs.extend(next_outs)
                number_outs_per_order.append(len(next_outs))
            else:
                outs = next_outs

        _ = res.item(
            0
        )  # call this on curr to compute all postponed outeputs of the potentially corresponding fx graph, this will cimpute all outputs at once

        # make out independent of the computation graph i.e. copy the data into fresh arrays
        var final_outs = List[Array]()
        for out in outs:
            var copy_out = Array(out[].shape())
            var out_dual = out[].graph_dual() if out[].has_fxgraph() else out[]
            execute_copy_raw(
                out_dual.data(),
                copy_out.data(),
                out_dual.array_shape(),
                out_dual.is_complex(),
            )
            final_outs.append(copy_out)

        # clean up
        if self.optimize_jit:
            captured_graph[].zero_data()
            captured_graph[].reset_data_and_shapes_to_uncomputed()

        # build out based on the order of differentiation and if we want to keep intermediate outs
        if len(final_outs) == 1:
            return final_outs[0]
        elif self.keep_intermediate_outs:
            var out = List[List[Array]]()
            var idx = 0
            for order in range(self.order_of_differentiation + 1):
                var tmp = List[Array]()
                for _ in range(number_outs_per_order[order]):
                    tmp.append(final_outs[idx])
                    idx += 1
                out.append(tmp)
            return out
        else:
            return final_outs
