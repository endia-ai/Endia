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
from endia.compile import Callable
from endia.utils import top_order_rec
from utils import Variant


fn backward(arg: Array, create_graph: Bool) raises:
    """Performs backward propagation on the given Array."""
    jacrev(arg, create_graph)


fn jacrev(arg: Array, create_graph: Bool) raises:
    """Computes the reverse-mode Jacobian for the given Array."""
    var out = arg

    reset_node_id_recursive(out)
    var trace = List[Array]()
    top_order_rec(out, trace)

    var dims = arg.shape()[arg.ndim() - 1]
    if not out.has_grad():
        var last_grad = reshape(eye(out.size()), out.shape() + out.shape()) if (
            out.ndim() != 1 or dims != 1
        ) else ones(dims)
        out.grad_(last_grad)

    for i in range(len(trace) - 1, -1, -1):
        var curr = trace[i]
        var primals = curr.args()

        if primals.size == 0:
            continue

        var vjp = curr.vjp()
        var grad = curr.grad()
        var primals_grads = vjp(primals, grad, curr)

        for j in range(len(primals)):
            var primal = primals[j]
            if primal.requires_grad():
                var primal_grad = primals_grads[j]
                if primal.has_grad():
                    primal_grad = add(primal_grad, primal.grad())
                primal.grad_(primal_grad)

                if primal.has_fxgraph():
                    if primal.is_breakpoint():
                        primal_grad.postpone_as_grpah_output()

                if not create_graph:
                    var primal_grad = primal.grad()
                    primal_grad.clear_args()

    reset_node_id_recursive(out)


fn grad(
    outs: List[Array],
    inputs: List[Array],
    retain_grads: Bool = True,
    create_graph: Bool = False,
) raises -> List[Array]:
    """Computes gradients of outputs with respect to inputs."""
    # for i in range(len(outs)):
    #     var out = outs[i]
    #     remove_grad_rec(out)
    for i in range(len(inputs)):
        var input = inputs[i]
        remove_grad_rec(input)
    for i in range(len(outs)):
        var out = outs[i]
        out.backward(create_graph=create_graph)
    var final_outs = List[Array]()
    for i in range(len(inputs)):
        var input = inputs[i]
        var gradient = input.grad()
        if not create_graph:
            gradient.clear_args()
            gradient.remove_grad()
        if not retain_grads:
            input.remove_grad()
        final_outs.append(gradient)

    return final_outs


fn grad(f: Callable, argnums: List[Int] = List(-1)) raises -> Callable:
    """Computes the gradient of a Callable function with respect to specified arguments.
    """
    var existing_argnums = f.argnums
    existing_argnums.append(argnums)
    return Callable(
        f.func,
        existing_argnums,
        f.order_of_differentiation + 1,
        False,
        False,
    )


fn grad(
    f: fn (List[Array]) raises -> Array, argnums: List[Int] = List(-1)
) raises -> Callable:
    """Computes the gradient of a function that takes a list of Arrays and returns an Array.
    """
    return Callable(f, argnums, 1, False, False)


fn grad(
    f: fn (Array) raises -> Array, argnums: List[Int] = List(-1)
) raises -> Callable:
    """Computes the gradient of a function that takes a single Array and returns an Array.
    """
    return Callable(f, argnums, 1, False, False)


fn jacobian(f: Callable, argnums: List[Int] = List(-1)) raises -> Callable:
    """Computes the Jacobian of a Callable function with respect to specified arguments.
    """
    return grad(f, argnums)


fn jacobian(
    f: fn (List[Array]) raises -> Array, argnums: List[Int] = List(-1)
) raises -> Callable:
    """Computes the Jacobian of a function that takes a list of Arrays and returns an Array.
    """
    return grad(f, argnums)


fn jacobian(
    f: fn (Array) raises -> Array, argnums: List[Int] = List(-1)
) raises -> Callable:
    """Computes the Jacobian of a function that takes a single Array and returns an Array.
    """
    return grad(f, argnums)


fn hessian(f: Callable, argnums: List[Int] = List(-1)) raises -> Callable:
    """Computes the Hessian of a Callable function with respect to specified arguments.
    """
    var f_grad = grad(f, argnums)
    return grad(f_grad, argnums)


fn hessian(
    f: fn (List[Array]) raises -> Array, argnums: List[Int] = List(-1)
) raises -> Callable:
    """Computes the Hessian of a function that takes a list of Arrays and returns an Array.
    """
    var f_grad = grad(f, argnums)
    return grad(f_grad, argnums)


fn hessian(
    f: fn (Array) raises -> Array, argnums: List[Int] = List(-1)
) raises -> Callable:
    """Computes the Hessian of a function that takes a single Array and returns an Array.
    """
    var f_grad = grad(f, argnums)
    return grad(f_grad, argnums)


fn value_and_grad(
    arg: Variant[Callable, fn (List[Array]) raises -> Array],
    argnums: List[Int] = List(-1),
) raises -> Callable:
    """Computes both the value and gradient of a function or Callable with respect to specified arguments.
    """
    var a = arg
    if arg.isa[Callable]():
        var _a = a.unsafe_take[Callable]()
        var existing_argnums = _a.argnums
        existing_argnums.append(argnums)
        return Callable(
            _a.func,
            existing_argnums,
            _a.order_of_differentiation + 1,
            False,
            True,
        )
    else:
        var _f = a.unsafe_take[fn (List[Array]) raises -> Array]()
        return Callable(_f, argnums, 1, False, True)


fn jacobian(f: Callable, args: List[Array]) raises -> List[Array]:
    """Computes the Jacobian of a Callable function with respect to given arguments.
    """
    var f_jac = grad(f, List(-1))
    return f_jac(args)[List[Array]]


fn jacobian(
    f: fn (List[Array]) raises -> Array, args: List[Array]
) raises -> List[Array]:
    """Computes the Jacobian of a function that takes a list of Arrays with respect to given arguments.
    """
    var f_jac = grad(f, List(-1))
    return f_jac(args)[List[Array]]


fn jacobian(f: fn (Array) raises -> Array, arg: Array) raises -> Array:
    """Computes the Jacobian of a function that takes a single Array with respect to the given argument.
    """
    var f_jac = grad(f, List(-1))
    return f_jac(arg)[Array]


fn hessian(f: Callable, args: List[Array]) raises -> List[Array]:
    """Computes the Hessian of a Callable function with respect to given arguments.
    """
    var f_jes = hessian(f, List(-1))
    return f_jes(args)[List[Array]]


fn hessian(
    f: fn (List[Array]) raises -> Array, args: List[Array]
) raises -> List[Array]:
    """Computes the Hessian of a function that takes a list of Arrays with respect to given arguments.
    """
    var f_jes = hessian(f, List(-1))
    return f_jes(args)[List[Array]]


fn hessian(f: fn (Array) raises -> Array, arg: Array) raises -> Array:
    """Computes the Hessian of a function that takes a single Array with respect to the given argument.
    """
    var f_jes = hessian(f, List(-1))
    return f_jes(arg)[Array]
