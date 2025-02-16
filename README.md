# Welcome to Endia 24.6

**Endia** is a Machine Learning Framework written in pure Mojo, featuring:

- **AD**: Compute derivatives of arbitrary order for training Neural Networks and beyond.
- **Signal Processing:** Complex Numbers, Fourier Transforms, Convolution, and more.
- **JIT Compilation:** Leverage the [MAX Engine](https://www.modular.com/) to speed up your code.
- **Dual API:** Choose between a PyTorch-like imperative or a JAX-like functional interface.

Checkout the [Endia documentation](https://endia.vercel) for more information.

## Usage

**Prerequisites: [Mojo 24.6](https://docs.modular.com/mojo/manual/get-started) 🔥 and [Magic](https://docs.modular.com/magic/)**

### Option 1: Install the package for usage in your project

- Add the Modular Community channel to your project's `.toml` file and install the package:

  ```shell
  magic project channel add "https://repo.prefix.dev/modular-community"
  magic add endia
  ```



### Option 2: Clone the repository and run all examples, tests and benchmarks

- **Clone the Repository**

    ```shell
    git clone https://github.com/endia-ai/Endia.git
    cd Endia
    magic shell
    ```

- **Run the Examples, Tests and Benchmarks**

    ```shell
    mojo run_all.mojo
    ````

    Recommended: Go to the `run_all.mojo` file and adjust the execution to your liking.

####

## A Simple Example - Computing Derivatives

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
from endia import Tensor, sum, arange
import endia.autograd.functional as F


# Define the function
def foo(x: Tensor) -> Tensor:
    return sum(x ** 2)

def main():
    # Initialize variable - requires_grad=True needed!
    x = arange(1.0, 4.0, requires_grad=True) # [1.0, 2.0, 3.0]

    # Compute result, first and second order derivatives
    y = foo(x)
    y.backward(create_graph=True)            
    dy_dx = x.grad()
    d2y_dx2 = F.grad(outs=sum(dy_dx), inputs=x)[0]

    # Print results
    print(y)        # 14.0
    print(dy_dx)    # [2.0, 4.0, 6.0]
    print(d2y_dx2)  # [2.0, 2.0, 2.0]
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
from endia import grad, jacobian
from endia.numpy import sum, arange, ndarray


def foo(x: ndarray) -> ndarray:
    return sum(x**2)


def main():
    # create Callables for the first and second order derivatives
    foo_jac = grad(foo)
    foo_hes = jacobian(foo_jac)

    x = arange(1.0, 4.0)       # [1.0, 2.0, 3.0]

    print(foo(x))              # 14.0
    print(foo_jac(x)[ndarray]) # [2.0, 4.0, 6.0]
    print(foo_hes(x)[ndarray]) # [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]
```

*And there is so much more! Endia can handle complex valued functions, can perform both forward and reverse-mode automatic differentiation, it even has a builtin JIT compiler to make things go brrr. Explore the full **list of features** in the [documentation](https://endia.vercel).*

## Contributing

Contributions to Endia are welcome! If you'd like to contribute, please follow the contribution guidelines in the [CONTRIBUTING.md](https://github.com/endia-ai/Endia/blob/main/CONTRIBUTING.md) file in the repository.

## License

Endia is licensed under the [Apache-2.0 license with LLVM Exeptions](https://github.com/endia-ai/Endia/blob/main/LICENSE).

## Happy Coding! 🚀

<div align="center" style="max-width: 1000px; margin: auto;">
  <img src="./assets/title_image.png" alt="Endia Title Image" style="max-width: 100%;" />
</div>

### 

![CodeQL](https://github.com/endia-ai/Endia/workflows/CodeQL/badge.svg)
