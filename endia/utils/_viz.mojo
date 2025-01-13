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
import os
from python import Python, PythonObject


fn graph_to_json(args: List[Array]) raises -> String:
    for arg in args:
        var out = arg[]
        reset_node_id_recursive(out)

    var trace = List[Array]()

    for arg in args:
        var out = arg[]
        top_order_rec(out, trace)

    var json_str: String = "{"
    json_str += "\n"
    json_str += '   "nodes": [\n'
    for i in range(len(trace)):
        var curr = trace[i]
        json_str += "       {\n"
        json_str += (
            '           "type": '
            + str('"')
            + str(curr.name())
            + str('"')
            + ",\n"
        )
        json_str += '           "id": ' + str(curr.id()) + ",\n"
        json_str += '           "is_view": '
        if curr.is_view():
            json_str += "1,\n"
        else:
            json_str += "0,\n"
        json_str += '           "shape": ['
        for j in range(len(curr.shape())):
            json_str += str(curr.shape()[j])
            if j != len(curr.shape()) - 1:
                json_str += ", "
        json_str += "],\n"
        json_str += "           " + '"args": ['
        for j in range(len(curr.args())):
            var arg = curr.args()[j]
            json_str += str(arg.id())
            if j != len(curr.args()) - 1:
                json_str += ", "
        json_str += "],\n"
        json_str += '           "grad": '
        if curr.has_grad():
            json_str += str(curr.grad().id())
        else:
            json_str += "null"
        json_str += "\n"
        json_str += "       }"
        if i != len(trace) - 1:
            json_str += ", "
        json_str += "\n"
    json_str += "   ]\n"
    json_str += "}"
    json_str += "\n"

    return json_str


fn write_graph_to_json(
    arg: Array, filename: String = "computation_graph.json"
) raises:
    """
    Write the computation graph of the given list of arrays to a json file.
    """
    var file = open(filename, "w")
    file.write(graph_to_json(arg))
    file.close()


fn visualize_graph(arg: Array, filename: String = "computation_graph") raises:
    """
    Visualize the computation graph of the given list of arrays using graphviz.
    """
    var graphviz: PythonObject
    try:
        graphviz = Python.import_module("graphviz")
    except:
        raise "\nGraphviz not found while running examples. Please install graphviz via: \n\033[92m magic add --pypi \"graphviz\" \033[0m"

    var json = Python.import_module("json")
    var Digraph = graphviz.Digraph

    var graph_data = json.loads(graph_to_json(arg))
    var dot = Digraph(comment=filename)
    dot.attr(rankdir="TB")

    # Set graph background to black
    dot.attr(bgcolor="white")

    # Change node attributes for better visibility on black background
    dot.attr(
        "node",
        shape="box",
        style="rounded,filled",
        roundedcorners="0.03",
        fontcolor="black",
    )

    for node in graph_data["nodes"]:
        var node_color = "white"  # Dark gray for standard nodes
        if len(node["args"]) == 0:
            node_color = "#b1d8fa"  # Steel blue for input nodes
        elif node["is_view"] == 1:
            node_color = "#ffc2a3"  # Dim gray for view nodes

        var attrs: String = " " + str(node["id"]) + ', "' + str(
            node["type"]
        ) + '", '
        var shape = node["shape"]
        attrs += "("
        for j in range(len(shape)):
            attrs += str(shape[j])
            if j != len(shape) - 1:
                attrs += ","
        attrs += ") "

        dot.node(str(node["id"]), attrs, fillcolor=node_color)

        for arg in node["args"]:
            dot.edge(str(arg), str(node["id"]), color="black")

        # Add grad edge if exists
        if node["grad"] is not None:
            dot.edge(
                str(node["id"]),
                str(node["grad"]),
                color="#FF6347",  # Tomato red for gradient edges
                style="dashed",
                constraint="false",
            )

    # Use render method with cleanup=True to avoid intermediate files
    dot.render(filename, format="png", cleanup=True)
