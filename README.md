<!-- <div align="center">
  <img src="./assets/titleimage.png" alt="Title Image" />
</div> -->

<div align="center">
  <img src="./assets/endia_stack_concept.png" alt="Endia Stack concept Image"/>
</div>

###

**Endia** is a dynamic Array library for Scientific Computing, similar to PyTorch, Numpy and JAX, but made in 7k lines of pure Mojo. It offers:

- **Automatic differentiation**: Compute derivatives of arbitrary order.
- **Complex number support:** Use Endia for advanced scientific applications.
- **Dual API:** Choose between a PyTorch-like imperative or a JAX-like functional interface.
- **JIT Compilation:** Leverage MAX to speed up training and inference.

<div align="center">
  
  [Website] | [Docs] | [Getting Started]

  [Website]: https://endia.vercel.app/
  [Docs]: https://endia.vercel.app/docs/array
  [Getting Started]: https://endia.vercel.app/docs/get_started

</div>

## Installation

1. **Install [Mojo and MAX](https://docs.modular.com/max/install)** 🔥 (v24.4)

2. **Clone the repository**: Choose one of the following options to clone the repository:


    ```bash
    git clone https://github.com/endia-org/Endia.git
    cd Endia
    ```

    If you aim to use the nightly (development) version, switch to the `nightly` branch:

    ```bash
    git checkout nightly
    ```

3. **Set Up Environment**:

    ```bash
    chmod +x setup.sh
    ./setup.sh
    ```

    Required dependencies: `torch`, `numpy`, `graphviz`. These will be installed automatically by the setup script.



## A tiny example

In this guide, we'll demonstrate how to compute the **value**, **gradient**, and the **Hessian** (i.e. the second-order derivative) of a simple function. First by using Endia's Pytorch-like API and then by using a more Jax-like functional API. In both examples, we initially define a function **foo** that takes an array and returns the sum of the squares of its elements.

### The **Pytorch** way

<!-- markdownlint-disable MD033 -->
<p align="center">
  <a href="https://pytorch.org/docs/stable/index.html">
    <img src="assets/pytorch_logo.png" alt="Endia Logo" width="40">
  </a>
</p>

When using Endia's imperative (PyTorch-like) interface, we compute the gradient of a function by calling the **backward** method on the function's output. This imperative style requires explicit management of the computational graph, including setting `requires_grad=True` for the input arrays (i.e. leaf nodes) and using `create_graph=True` in the backward method when computing higher-order derivatives.

```python
import endia as nd 


# Define the function
def foo(x: nd.Array) -> nd.Array:
    return nd.sum(x ** 2)


# Initialize variable - requires_grad=True needed!
x = arange(1.0, 4.0, requires_grad=True) # [1.0, 2.0, 3.0]

# Compute result, first and second order derivatives
y = foo(x)
y.backward(create_graph=True)            
dy_dx = x.grad()
d2y_dx2 = nd.grad(outs=nd.sum(dy_dx), inputs=x)[0]

# Print results
print(y)        # 14.0
print(dy_dx)    # [2.0, 4.0, 6.0]
print(dy2_dx2)  # [2.0, 2.0, 2.0]
```

### The **JAX** way

<!-- markdownlint-disable MD033 -->
<p align="center">
  <a href="https://jax.readthedocs.io/en/latest/quickstart.html">
    <img src="assets/jax_logo.png" alt="Endia Logo" width="65">
  </a>
</p>

When using Endia's functional (JAX-like) interface, the computational graph is handled implicitly. By calling the `grad` or `jacobian` function on foo, we create a `Callable` which computes the full Jacobian matrix. This `Callable` can be passed to the `grad` or `jacobian` function again to compute higher-order derivatives.

```python
from endia import grad, jacobian, sum, arange, Array


def foo(x: Array) -> Array:
    return sum(x**2)


# create Callables for the first and second order derivatives
foo_jac = grad(foo)
foo_hes = jacobian(foo_jac)

x = arange(1.0, 4.0)      # [1.0, 2.0, 3.0]

print(foo(x))             # 14.0
print(foo_jac(x)[Array])  # [2.0, 4.0, 6.0]
print(foo_hes(x)[Array])  # [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]
```

*And there is so much more! Endia can handle complex valued functions, can perform both forward and reverse-mode automatic differentiation, it even has a builtin JIT compiler to make things go brrr. Explore the full **list of features** in the [documentation](https://endia.org).*

## Our Mission

- 🧠 **Advance AI & Scientific Computing:** Push boundaries with clear and understandable algorithms
- 🚀 **Mojo-Powered Clarity:** High-performance open-source code that remains readable and pythonic through and through
- 📐 **Explainability:** Prioritize clarity and educational value over exhaustive features

## Contributing

Contributions to Endia are welcome! If you'd like to contribute, please follow the contribution guidelines in the [CONTRIBUTING.md](https://github.com/endia-org/Endia/blob/main/CONTRIBUTING.md) file in the repository.

## Citation

If you use Endia in your research or project, please cite it as follows:

```bibtex
@software{Fehrenbach_Endia_2024,
  author = {Fehrenbach, Tillmann},
  license = {Apache-2.0 with LLVM Exceptions},
  doi = {10.5281/zenodo.12810766},
  month = jul,
  title = {{Endia}},
  url = {https://github.com/endia-org/Endia},
  version = {24.4.2},
  year = {2024}
}
```

## License

Endia is licensed under the [Apache-2.0 license with LLVM Excpetions](https://github.com/endia-org/Endia/blob/main/LICENSE).
