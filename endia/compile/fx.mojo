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

from endia import Array, Node, ArrayShape, ShapeNode
from compile import *
from endia.functional._utils import (
    contiguous,
    op_array,
    setup_array_shape,
    setup_shape_and_data,
    execute_copy_raw,
)
from collections import Dict
from max.engine import Model
from .max_utils import build_model, execute_model


# @value
struct FxSubgraph(CollectionElement):
    """
    FxSubgraph represents a functionally pure subgraph within a larger computation graph. It facilitates optimization
    and efficient execution of subgraphs by caching and reusing optimized computations.
    """

    var traversing_arrays: List[Array]
    var inputs: List[Array]
    var outputs: List[Array]
    var max_model: List[ArcPointer[Model]]
    var compile_with_MAX: Bool

    fn __init__(
        inout self,
        compile_with_MAX: Bool,
        traversing_arrays: List[Array] = List[Array](),
    ):
        self.traversing_arrays = traversing_arrays
        self.inputs = List[Array]()
        self.outputs = List[Array]()
        self.max_model = List[ArcPointer[Model]]()
        self.compile_with_MAX = compile_with_MAX

    fn __copyinit__(inout self, other: FxSubgraph):
        self.traversing_arrays = other.traversing_arrays
        self.inputs = other.inputs
        self.outputs = other.outputs
        self.max_model = other.max_model
        self.compile_with_MAX = other.compile_with_MAX

    fn __moveinit__(inout self, owned other: FxSubgraph):
        self.traversing_arrays = other.traversing_arrays^
        self.inputs = other.inputs^
        self.outputs = other.outputs^
        self.max_model = other.max_model^
        self.compile_with_MAX = other.compile_with_MAX

    fn append(inout self, arr: Array):
        self.traversing_arrays.append(arr)

    fn setup_inputs_and_outputs(inout self) raises:
        var is_input = Dict[Int, Bool]()
        var is_output = Dict[Int, Bool]()
        for graph_node in self.traversing_arrays:
            var id_in_graph = graph_node[].id_in_graph()
            is_input[
                id_in_graph
            ] = False  # if a node has either no args at all or no args in the subgraph, then it is an input
            is_output[
                id_in_graph
            ] = True  # if a node has no users in the subgraph, then it is an output

        for graph_node in self.traversing_arrays:
            var node_id = graph_node[].id_in_graph()
            var args = graph_node[].args()
            var has_no_args_in_subgraph = True
            for arg in args:
                var arg_id = arg[].id_in_graph()
                is_output[arg_id] = False
            for arg in args:
                var arg_id = arg[].id_in_graph()
                if arg_id in is_input:
                    has_no_args_in_subgraph = False
                    break
            if has_no_args_in_subgraph:
                is_input[node_id] = True
            else:
                is_input[node_id] = False

        for graph_node in self.traversing_arrays:
            var node_id = graph_node[].id_in_graph()
            if is_input[node_id]:
                self.inputs.append(graph_node[])
            if is_output[node_id]:
                self.outputs.append(graph_node[])

    fn execute(inout self) raises:
        if self.compile_with_MAX:
            # Use a compiled subgraph with the MAX engine and execute it

            # needed for the max graph building: nodes and their ids need to be unique and in order
            for i in range(len(self.traversing_arrays)):
                self.traversing_arrays[i].id_(i)

            # Do once: setup the data and shape of the output nodes, then build the max model
            if len(self.max_model) == 0:
                self.setup_inputs_and_outputs()
                for output in self.outputs:
                    var array_shape = output[].array_shape()
                    compute_shape(array_shape, True)
                    var base = output[].base()
                    setup_shape_and_data(base)

                self.max_model.append(
                    ArcPointer(
                        build_model(
                            self.inputs, self.outputs, self.traversing_arrays
                        )
                    )
                )

            # execute the max model with the current inputs and copy the outpus back into the callable trace
            var max_model = self.max_model[0]
            var new_outputs = execute_model(
                self.inputs, self.outputs, max_model[]
            )

            for i in range(len(new_outputs)):
                var dst = self.outputs[i]
                var src = new_outputs[i]
                for j in range(dst.size()):
                    dst.store(j, src.load(j))

        else:
            # use Endia's default execution
            var graph = self.traversing_arrays[0].graph()

            for graph_node in self.traversing_arrays:
                var curr = graph_node[]
                if len(curr.args()) == 0:
                    continue
                var id_in_graph = curr.id_in_graph()
                var fwd = curr.fwd()
                var args = curr.args()
                fwd(curr, args)
                # graph[].trace[id_in_graph].is_computed = True

        # reset the node ids and set the is_computed flag to True
        for curr in self.traversing_arrays:
            curr[].id_(-1)
            curr[].is_graph_node_computed_(True)

    fn print(self) raises:
        """
        Print the subgraph in a human readable table like format. It will show the flow of the computation from the
        top to the bottom, and also will show the direct dependencies (args) and other metadata such as the shape, stride,
        storage_offset and the requires_grad flag.
        """
        # print an IR like tabular representation of the subgraph
        var opcode_reference = "opcode          "
        var name_reference = "name             "
        var target_reference = "target        "
        var args_reference = "args                     "
        var kwargs_reference = "kwargs      "
        var shape_reference = "[shape, stride, storage_offset]"
        var header: String = opcode_reference + " | " + name_reference + " | " + target_reference + " | " + args_reference + " | " + kwargs_reference + " | " + shape_reference + " | "

        var header_sub = String("_") * len(opcode_reference) + " | " + String(
            "_"
        ) * len(name_reference) + " | " + String("_") * len(
            target_reference
        ) + " | " + String(
            "_"
        ) * len(
            args_reference
        ) + " | " + String(
            "_"
        ) * len(
            kwargs_reference
        ) + " | " + String(
            "_"
        ) * len(
            shape_reference
        )
        print(header)
        print(header_sub)

        for i in range(len(self.traversing_arrays)):
            var curr = self.traversing_arrays[i]

            var opcode: String = "placeholder"
            if len(curr.args()) > 0:
                opcode = "call_function"
            if i == len(self.traversing_arrays) - 1:
                opcode = "out"
            opcode += String(" ") * (len(opcode_reference) - len(opcode))
            var name: String = str(curr.id_in_graph())
            name += "_"
            name += str(curr.name())
            name += String(" ") * (len(name_reference) - len(name))

            var target = curr.name()
            target += String(" ") * (len(target_reference) - len(target))

            var args: String = ""
            for arg in curr.args():
                if arg[].has_fxgraph():
                    args += str(arg[].id_in_graph())
                else:
                    args += str(-1)
                args += "_"
                args += arg[].name()
                args += ", "
            if len(curr.args()) == 0:
                args = "{}"
            args += String(" ") * (len(args_reference) - len(args))

            var kwargs: String = ""
            for kwarg in curr.kwargs():
                if kwarg[].has_fxgraph():
                    kwargs += str(kwarg[].id_in_graph())
                else:
                    kwargs += str(-1)
                kwargs += "_"
                kwargs += kwarg[].name()
                kwargs += ", "
            if len(curr.kwargs()) == 0:
                kwargs = "{}"
            kwargs += String(" ") * (len(kwargs_reference) - len(kwargs))

            var shape = curr.shape()
            var stride = curr.stride()
            var storage_offset = curr.storage_offset()
            var shape_str: String = "["
            for i in range(len(shape)):
                shape_str += str(shape[i])
                if i < len(shape) - 1:
                    shape_str += "x"
            shape_str += ", "
            for i in range(len(stride)):
                shape_str += str(stride[i])
                if i < len(stride) - 1:
                    shape_str += "x"
            shape_str += ", "
            shape_str += str(storage_offset)
            shape_str += "]"
            shape_str += String(" ") * (len(shape_reference) - len(shape_str))

            print(
                opcode,
                "|",
                name,
                "|",
                target,
                "|",
                args,
                "|",
                kwargs,
                "|",
                shape_str,
            )

    fn IR(self) raises -> String:
        """
        Get an IR like code representation of the subgraph. As of right perf_counter this has perf_counter real functionality, but eventually this
        IR string should become a valid MLIR code representation of the subgraph, which can be compiled and optimized by the MLIR.
        """
        # create an IR like code representation of the subgraph
        var IR: String = "\n"
        var IR_header: String = "func @fx_subgraph("
        var IR_body: String = ""
        var out_name: String = ""
        var out_shape: String = ""

        for i in range(len(self.traversing_arrays)):
            var curr = self.traversing_arrays[i]

            var opcode: String = "placeholder"
            if len(curr.args()) > 0:
                opcode = "call_function"
            if i == len(self.traversing_arrays) - 1:
                opcode = "out"

            var name: String = str(curr.id_in_graph())
            name += "_"
            name += str(curr.name())

            var target = curr.name()

            var args: String = ""
            for j in range(len(curr.args())):
                var arg = curr.args()[j]
                args += "%"
                if arg.has_fxgraph():
                    args += str(arg.id_in_graph())
                else:
                    args += str(-1)
                args += "_"
                args += arg.name()
                if j < len(curr.args()) - 1:
                    args += ", "

            if len(curr.args()) == 0:
                args = "{}"

            var kwargs: String = ""
            for j in range(len(curr.kwargs())):
                var kwarg = curr.kwargs()[j]
                if kwarg.has_fxgraph():
                    kwargs += str(kwarg.id_in_graph())
                else:
                    kwargs += str(-1)
                kwargs += "_"
                kwargs += kwarg.name()
                if j < len(curr.kwargs()) - 1:
                    kwargs += ", "
            if len(curr.kwargs()) == 0:
                kwargs = "{}"

            var shape = curr.shape()
            # var stride = curr.stride()
            # var storage_offset = curr.storage_offset()
            var shape_str: String = ""
            for i in range(len(shape)):
                shape_str += str(shape[i])
                if i < len(shape) - 1:
                    shape_str += "x"
            shape_str += "xf32"

            if opcode == "placeholder":
                IR_header += "%"
                IR_header += name
                IR_header += ": tensor<"
                IR_header += shape_str
                IR_header += ">, "
            elif opcode == "call_function" or opcode == "out":
                IR_body += "    %"
                IR_body += name
                IR_body += " = @"
                IR_body += target
                IR_body += " "
                IR_body += args
                IR_body += " -> "
                IR_body += " tensor<"
                IR_body += shape_str
                IR_body += ">\n"
                if opcode == "out":
                    out_name = name
                    out_shape = shape_str
            else:
                pass

        IR_header += ") -> tensor<"
        IR_header += out_shape + "> {\n"

        IR += IR_header
        IR += IR_body

        IR += "    return %"
        IR += out_name + " : " + "tensor<" + out_shape + ">\n"

        IR += "}\n"
        return IR


@value
struct FxGraphNode(CollectionElement):
    """
    FxGraphNode is a lightweight dual representation of an Array (or Node) within a traced function. It serves as a
    bookkeeping structure to facilitate tracing, caching, and optimization of computation graphs.
    """

    var array_in_graph: Array
    var name: String
    var branch_to_idx: Int
    var is_breakpoint: Bool
    var dependencies: Int
    var sub_graph: List[ArcPointer[FxSubgraph]]
    var tmp_id_in_subgraph: Int
    var jvp_derivatives: List[Array]
    var is_computed: Bool
    var id: Int

    fn __init__(
        inout self, name: String, branch_to_idx: Int, array_in_graph: Array
    ):
        self.array_in_graph = array_in_graph
        self.name = name
        self.branch_to_idx = branch_to_idx
        self.is_breakpoint = False
        self.dependencies = 0
        self.sub_graph = List[ArcPointer[FxSubgraph]]()
        self.tmp_id_in_subgraph = -1
        self.jvp_derivatives = List[Array]()
        self.is_computed = False
        self.id = -1

    fn print(self, storage_offset: String = "") raises:
        print(
            storage_offset,
            self.name
            + (
                " -> potentially jump to "
                + str(self.branch_to_idx) if self.branch_to_idx
                != -1 else ""
            ),
        )
        print(str(self.array_in_graph))

    fn subgraph(self) raises -> FxSubgraph:
        if not self.sub_graph:
            raise "Subgraph not yet computed"
        return self.sub_graph[0][]


@value
struct FxGraph:
    """
    FxGraph is a data structure that holds the traced operations and computation graph of a function. It facilitates
    Just-In-Time (JIT) compilation, optimization, and caching of subgraphs within the computation graph.
    """

    var trace: List[FxGraphNode]
    var curr_idx: Int
    var postponed_outputs: List[Int]
    var compile_with_MAX: Bool

    fn __init__(inout self, compile_with_MAX: Bool):
        self.trace = List[FxGraphNode]()
        self.curr_idx = 0
        self.postponed_outputs = List[Int]()
        self.compile_with_MAX = compile_with_MAX

    fn op_arrayeration(inout self, inout arr: Array) raises:
        if arr.id() == -2:
            raise "Error: This is a test error."
        var name = arr.name()
        # print(name, self.curr_idx)
        if self.curr_idx >= len(self.trace):
            # print("     registering new entry", arr.name())
            # array is not initalized in fxgraph, but the array shape subgraph is being setup already, still dynamically computed when data is specified though!
            var id_in_graph = len(self.trace)
            arr.id_in_graph_(id_in_graph)
            self.trace.append(FxGraphNode(name, -1, arr))
            self.curr_idx = len(self.trace)
        else:
            var registered_op = self.trace[self.curr_idx]
            if registered_op.name == name:
                # print("     same operation", arr.name())
                arr.id_in_graph_(self.curr_idx)
                self.curr_idx += 1
                return
            else:
                if registered_op.branch_to_idx == -1:
                    # print("     setup new jump at the end of the trace", arr.name())
                    self.trace[self.curr_idx] = FxGraphNode(
                        registered_op.name,
                        len(self.trace),
                        registered_op.array_in_graph,
                    )
                    self.curr_idx = len(self.trace)
                    self.op_arrayeration(arr)
                else:
                    # print("     jumping to registered alternative operation", arr.name())
                    self.curr_idx = registered_op.branch_to_idx
                    self.op_arrayeration(arr)
        # print("->",name, arr.id_in_graph(), arr.has_fxgraph(), self.curr_idx, len(self.trace))

    fn reset_data_and_shapes_to_uncomputed(inout self) raises:
        for graph_node in self.trace:
            var arr = graph_node[].array_in_graph
            if len(arr.args()) == 0:
                continue
            # var array_shape = arr.array_shape()
            # array_shape.shape_node[].is_computed = False
            arr.is_computed_(False)
            arr.is_graph_node_computed_(False)

    fn setup_grads(inout self) raises:
        for graph_node in self.trace:
            var arr = graph_node[].array_in_graph
            var requires_grad = arr.requires_grad()
            if requires_grad:
                arr.grad_(Array(arr.shape(), requires_grad=True))

    fn zero_data(inout self) raises:
        for graph_node in self.trace:
            var array_in_graph = graph_node[].array_in_graph
            # self.trace[array_in_graph.id_in_graph()].is_computed = False
            array_in_graph.is_graph_node_computed_(False)
            if not array_in_graph.is_view() and len(array_in_graph.args()) == 0:
                var data = array_in_graph.data()
                var size = graph_node[].array_in_graph.size()
                memset_zero(data, size)

    fn subgraph(inout self, compile_with_MAX: Bool) raises -> ArcPointer[FxSubgraph]:
        var subgraph_list = List[Array]()
        # var curr = self.trace[breakpoint_id].array_in_graph
        # reset_node_id_recursive(curr)

        for graph_id in self.postponed_outputs:
            var graph_node = self.trace[graph_id[]]
            var postponed_output = graph_node.array_in_graph
            reset_node_id_recursive(postponed_output)

        # print(len(self.postponed_outputs))

        for graph_id in self.postponed_outputs:
            var graph_node = self.trace[graph_id[]]
            var postponed_output = graph_node.array_in_graph
            top_order_subgraph_rec(postponed_output, subgraph_list)
            # self.trace[graph_id[]].is_breakpoint = True

        for graph_id in self.postponed_outputs:
            var graph_node = self.trace[graph_id[]]
            var postponed_output = graph_node.array_in_graph
            reset_node_id_recursive(postponed_output)

        self.postponed_outputs.clear()

        # adapt the trace of the subgraph such that all args/inputs are at the beginning:
        var subgraph_list_inputs = List[Array]()
        var subgraph_list_rest = List[Array]()

        for arr in subgraph_list:
            if arr[].name() == "arg":
                subgraph_list_inputs.append(arr[])
            else:
                subgraph_list_rest.append(arr[])

        var subgraph_final_trace = List[Array]()
        for arr in subgraph_list_inputs:
            subgraph_final_trace.append(arr[])
        for arr in subgraph_list_rest:
            subgraph_final_trace.append(arr[])

        # # print
        # for i in range(len(subgraph_final_trace)):
        #     print(subgraph_final_trace[i].name())

        # for arr in subgraph_final_trace:
        #     print(arr[].id(), arr[].name(), " -> [", end="")
        #     for arg in arr[].args():
        #         print(arg[].id(), arg[].name(), ", ", end="")
        #     print("]")

        return ArcPointer(FxSubgraph(compile_with_MAX, subgraph_final_trace))


fn top_order_subgraph_rec(inout curr: Array, inout trace: List[Array]) raises:
    # if curr.id() != -1 or curr.is_breakpoint() or curr.is_graph_node_computed():
    #     return
    for arg in curr.args():
        if (
            arg[].node[].id == -1
            and not curr.is_breakpoint()
            and not curr.is_graph_node_computed()
        ):
            var arg_deref = arg[]
            top_order_rec(arg_deref, trace)
    curr.id_(len(trace))
    trace.append(curr)
