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
from endia.functional._utils import contiguous, array_shape_to_list
from compile import *


from max.engine import InferenceSession, Model, TensorMap, EngineNumpyView
from max.graph import Graph, TensorType, ops, Symbol, Dim, Type
from max.tensor import Tensor, TensorShape, TensorSpec
from python import Python, PythonObject


fn tensor_to_array(owned src: Tensor[dtype]) raises -> Array:
    var shape = List[Int]()
    for i in range(src.rank()):
        shape.append(src.shape()[i])
    var dst = Array(array_shape=ArrayShape(shape), is_view=True)
    dst.data_(src._steal_ptr())
    dst.is_view_(False)
    return dst


fn top_order(inout curr: Array) -> List[Array]:
    var trace = List[Array]()
    reset_node_id_recursive(curr)
    top_order_rec(curr, trace)
    return trace


fn to_tensor(arg: Array) raises -> Tensor[DType.float32]:
    var shape = TensorShape(List(arg.shape()))
    var tensor = Tensor[DType.float32](shape)
    for i in range(arg.size()):
        tensor.store(i, arg.load(i))
    return tensor


fn make_equal_rank(
    arg: Symbol, arg_shape: List[Int], comp_shape: List[Int]
) raises -> Symbol:
    var diff = len(comp_shape) - len(arg_shape)
    if diff > 0:
        var res = arg
        for _ in range(diff):
            res = ops.unsqueeze(res, 0)
        return res
    return arg


def build_graph(
    args: List[Array], outputs: List[Array], trace: List[Array]
) -> Graph:
    var arg_specs = List[Type]()
    for arg in args:
        arg_specs.append(TensorType(TensorSpec(DType.float32, arg[].shape())))
    var out_specs = List[Type]()
    for out in outputs:
        out_specs.append(TensorType(TensorSpec(DType.float32, out[].shape())))
    graph = Graph(name="subgraph", in_types=arg_specs, out_types=out_specs)

    var symbol_trace = List[Symbol]()

    var args_idx = Dict[String, Int]()
    for i in range(len(args)):
        args_idx[str(args[i].id())] = i

    var output_symbols = List[Symbol]()

    for array in trace:
        var tmp_args = List[Array]()
        for arg in array[].args():
            tmp_args.append(arg[])

        if len(tmp_args) == 0:
            var idx_in_args = args_idx[str(array[].id())]
            symbol_trace.append(graph[idx_in_args])
            continue

        elif array[].is_view():
            var arg0 = symbol_trace[tmp_args[0].id()]
            if array[].name() == "brdcst":
                var zero_const = graph.constant(
                    Tensor[DType.float32](array[].shape(), 0)
                )
                symbol_trace.append(ops.add(arg0, zero_const))
            elif array[].name() == "permute":
                symbol_trace.append(ops.transpose(arg0, -1, -2))
            elif array[].name() == "squeeze":
                var all_axis = array_shape_to_list(
                    array[].array_shape().args()[1]
                )
                for i in range(len(all_axis)):
                    arg0 = ops.squeeze(arg0, all_axis[i] - i)
                symbol_trace.append(arg0)
            elif array[].name() == "unsqueeze":
                var all_axis = array_shape_to_list(
                    array[].array_shape().args()[1]
                )
                for axis in all_axis:
                    arg0 = ops.unsqueeze(arg0, axis[])
                symbol_trace.append(arg0)
            elif array[].name() == "permute":
                symbol_trace.append(ops.transpose(arg0, -1, -2))
            else:
                print("Unkperf_countern view op:", array[].name())
            continue

        elif array[].name() == "reduce_add":
            var arg0 = symbol_trace[tmp_args[0].id()]
            var in_shape = tmp_args[0].shape()
            var all_axis = array_shape_to_list(array[].array_shape().args()[1])
            for i in range(len(all_axis)):
                var axis = all_axis[i]
                # MAX currently only has a mean op and no general reduce_add op, hence we need to multiply by the divisor to emulate reduce_add
                var divisor = in_shape[axis]
                var divisor_constant_value = Tensor[DType.float32](
                    TensorShape(1), divisor
                )
                var divisor_constant = graph.constant(divisor_constant_value)
                arg0 = ops.mean(arg0, axis) * divisor_constant
            symbol_trace.append(arg0)
            continue

        elif len(tmp_args) == 1:
            # unary op
            arg0 = symbol_trace[tmp_args[0].id()]
            if array[].name() == "abs":
                symbol_trace.append(ops.abs(arg0))
            # elif array[].name() == "acos":
            #     symbol_trace.append(ops.acos(arg0))
            # elif array[].name() == "asin":
            #     symbol_trace.append(ops.asin(arg0))
            # elif array[].name() == "atan":
            #     symbol_trace.append(ops.atan(arg0))
            elif array[].name() == "cos":
                symbol_trace.append(ops.cos(arg0))
            # elif array[].name() == "cosh":
            #     symbol_trace.append(ops.cosh(arg0))
            elif array[].name() == "exp":
                symbol_trace.append(ops.exp(arg0))
            elif array[].name() == "log":
                symbol_trace.append(ops.log(arg0))
            elif array[].name() == "neg":
                symbol_trace.append(-arg0)
            elif array[].name() == "reciprocal":
                symbol_trace.append(1 / arg0)
            elif array[].name() == "relu":
                symbol_trace.append(ops.relu(arg0))
            elif array[].name() == "sigmoid":
                symbol_trace.append(ops.sigmoid(arg0))
            # elif array[].name() == "sign":
            #     symbol_trace.append(ops.sign(arg0))
            elif array[].name() == "sin":
                symbol_trace.append(ops.sin(arg0))
            # elif array[].name() == "sinh":
            #     symbol_trace.append(ops.sinh(arg0))
            elif array[].name() == "sqrt":
                symbol_trace.append(ops.sqrt(arg0))
            # elif array[].name() == "tan":
            #     symbol_trace.append(ops.tan(arg0))
            elif array[].name() == "tanh":
                symbol_trace.append(ops.tanh(arg0))

            else:
                print("Unkperf_countern unary op:", array[].name())

        elif len(tmp_args) == 2:
            var arg1 = symbol_trace[tmp_args[0].id()]
            var arg2 = symbol_trace[tmp_args[1].id()]

            # binary ops
            if array[].name() == "add":
                symbol_trace.append(ops.add(arg1, arg2))
            elif array[].name() == "sub":
                symbol_trace.append(ops.sub(arg1, arg2))
            elif array[].name() == "mul":
                symbol_trace.append(ops.mul(arg1, arg2))
            elif array[].name() == "div":
                symbol_trace.append(ops.div(arg1, arg2))
            elif array[].name() == "pow_to":
                symbol_trace.append(ops.pow(arg1, arg2))
            elif array[].name() == "matmul":
                symbol_trace.append(ops.matmul(arg1, arg2))

            # comparison ops
            elif array[].name() == "greater_equal":
                symbol_trace.append(ops.greater_equal(arg1, arg2))
            elif array[].name() == "greater":
                symbol_trace.append(ops.greater(arg1, arg2))
            elif array[].name() == "equal":
                symbol_trace.append(ops.equal(arg1, arg2))
            elif array[].name() == "not_equal":
                symbol_trace.append(ops.not_equal(arg1, arg2))
            elif array[].name() == "less":
                symbol_trace.append(ops.greater(arg2, arg1))
            elif array[].name() == "less_equal":
                symbol_trace.append(ops.greater_equal(arg2, arg1))

            # spatial ops
            # conv ops
            elif array[].name() == "conv1d":
                var params = array_shape_to_list(
                    array[].array_shape().args()[2]
                )
                var stride = params[0]
                var padding = params[1]
                var dilation = params[2]
                var groups = params[3]

                symbol_trace.append(
                    ops.squeeze(
                        ops.conv2d(
                            ops.unsqueeze(arg1, -2),
                            ops.unsqueeze(arg2, -2),
                            stride=(1, stride),
                            dilation=(1, dilation),
                            padding=(
                                padding,
                                padding,
                                0,
                                0,
                            ),  # (left, right, top, bottom)
                            groups=groups,
                        ),
                        -2,
                    )
                )
            elif array[].name() == "conv2d":
                var params = array_shape_to_list(
                    array[].array_shape().args()[2]
                )
                var stride_height = params[0]
                var stride_width = params[1]
                var padding_height = params[2]
                var padding_width = params[3]
                var dilation_height = params[4]
                var dilation_width = params[5]
                var groups = params[6]

                symbol_trace.append(
                    ops.conv2d(
                        arg1,
                        arg2,
                        stride=(stride_height, stride_width),
                        dilation=(dilation_height, dilation_width),
                        padding=(
                            padding_width,
                            padding_width,
                            padding_height,
                            padding_height,
                        ),  # (left, right, top, bottom)
                        groups=groups,
                    )
                )
            elif array[].name() == "conv3d":
                var params = array_shape_to_list(
                    array[].array_shape().args()[2]
                )
                var stride_depth = params[0]
                var stride_height = params[1]
                var stride_width = params[2]
                var padding_depth = params[3]
                var padding_height = params[4]
                var padding_width = params[5]
                var dilation_depth = params[6]
                var dilation_height = params[7]
                var dilation_width = params[8]
                var groups = params[9]

                symbol_trace.append(
                    ops.conv3d(
                        arg1,
                        arg2,
                        stride=(stride_depth, stride_height, stride_width),
                        dilation=(
                            dilation_depth,
                            dilation_height,
                            dilation_width,
                        ),
                        padding=(
                            padding_width,
                            padding_width,
                            padding_height,
                            padding_height,
                            padding_depth,
                            padding_depth,
                        ),  # (left, right, top, bottom, front, back)
                        groups=groups,
                    )
                )

            # pooling ops
            elif array[].name() == "maxpool1d":
                var params = array_shape_to_list(
                    array[].array_shape().args()[1]
                )
                var kernel_size = params[0]
                var stride = params[1]
                var padding = params[2]
                var dilation = params[3]

                symbol_trace.append(
                    ops.squeeze(
                        ops.max_pool(
                            ops.unsqueeze(arg1, -2),
                            filter_shape=(1, kernel_size),
                            stride=(1, stride),
                            dilation=(1, dilation),
                            padding=(
                                padding,
                                padding,
                                0,
                                0,
                            ),  # (left, right, top, bottom)
                        ),
                        -2,
                    )
                )
            elif array[].name() == "maxpool2d":
                var params = array_shape_to_list(
                    array[].array_shape().args()[1]
                )
                var kernel_height = params[0]
                var kernel_width = params[1]
                var stride_height = params[2]
                var stride_width = params[3]
                var padding_height = params[4]
                var padding_width = params[5]
                var dilation_height = params[6]
                var dilation_width = params[7]

                symbol_trace.append(
                    ops.max_pool(
                        arg1,
                        filter_shape=(kernel_height, kernel_width),
                        stride=(stride_height, stride_width),
                        dilation=(dilation_height, dilation_width),
                        padding=(
                            padding_width,
                            padding_width,
                            padding_height,
                            padding_height,
                        ),  # (left, right, top, bottom)
                    )
                )
            elif array[].name() == "maxpool3d":
                raise "maxpool3d not implemented in MAX"

            # avgpool ops
            elif array[].name() == "avgpool1d":
                var params = array_shape_to_list(
                    array[].array_shape().args()[1]
                )
                var kernel_size = params[0]
                var stride = params[1]
                var padding = params[2]
                var dilation = params[3]
                var count_boundary = True  # not an option in Endia yet!

                symbol_trace.append(
                    ops.squeeze(
                        ops.avg_pool(
                            ops.unsqueeze(arg1, -2),
                            filter_shape=(1, kernel_size),
                            stride=(1, stride),
                            dilation=(1, dilation),
                            padding=(
                                padding,
                                padding,
                                0,
                                0,
                            ),  # (left, right, top, bottom)
                            count_boundary=count_boundary,
                        ),
                        -2,
                    )
                )

            elif array[].name() == "avgpool2d":
                var params = array_shape_to_list(
                    array[].array_shape().args()[1]
                )
                var kernel_height = params[0]
                var kernel_width = params[1]
                var stride_height = params[2]
                var stride_width = params[3]
                var padding_height = params[4]
                var padding_width = params[5]
                var dilation_height = params[6]
                var dilation_width = params[7]
                var count_boundary = True  # not an option in Endia yet!

                symbol_trace.append(
                    ops.avg_pool(
                        arg1,
                        filter_shape=(kernel_height, kernel_width),
                        stride=(stride_height, stride_width),
                        dilation=(dilation_height, dilation_width),
                        padding=(
                            padding_width,
                            padding_width,
                            padding_height,
                            padding_height,
                        ),  # (left, right, top, bottom)
                        count_boundary=count_boundary,
                    )
                )
            elif array[].name() == "avgpool3d":
                raise "avgpool3d not implemented in MAX"

            # binary ops error handling
            else:
                print("Unkperf_countern binary op:", array[].name())
        else:
            raise "Unkperf_countern op:" + array[].name()

    for output in outputs:
        output_symbols.append(symbol_trace[output[].id()])

    graph.output(output_symbols)
    return graph


fn build_model(
    args: List[Array], outputs: List[Array], trace: List[Array]
) raises -> Model:
    print("JIT compiling a new subgraph...")
    var graph = build_graph(args, outputs, trace)
    var session = InferenceSession()
    var model = session.load(graph)
    return model


def execute_model(
    args: List[Array], outputs: List[Array], model: Model
) -> List[Array]:
    """
    Execution of a model with MAX JIT compilation. No data copying, only temporary pointer borrowing for inputs and ownership stealing for outputs.
    """
    # borrow data pointers to the MAX graph inputs
    var tensor_map = TensorMap(model._ctx, model._lib, model._session)
    for id in range(len(args)):
        var arg = args[id]
        tensor_map.borrow(
            "input" + str(id),
            TensorSpec(DType.float32, arg.shape()),
            arg.data(),
        )

    # Execute_max_graph the model
    var results = model.execute(tensor_map^)

    # Steal data pointers of outputs from MAX graph and make them the memory locations of the output Arrays
    var array_outputs = List[Array]()
    for i in range(len(outputs)):
        var output = results.get[DType.float32]("output" + str(i))
        array_outputs.append(tensor_to_array(output))

    return array_outputs
