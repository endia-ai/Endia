# Array Operations

Array operations are operations that take one or more Arrays as input and return one or more Arrays as output. Among others, Array operations can be categorized into the following types: unary operations, binary operations and view operations.


# Build your own Custom Ops

Endia allows you to create **custom differentiable operations** 🔥, enabling you to:

1. Implement new operations not available in the endia core library
2. Optimize existing operations for better performance
3. Develop domain-specific operations for your unique needs

## Creating a Custom Operation - A Step-by-Step Guide

Let's walk through creating a custom elementwise multiplication operation, breaking it down into three key steps:

1. Implement the forward pass as a **low-level** function
2. Define the backward pass as a high-level **vector-Jacobian** product (vjp) function
3. **Register** the operation with Endia

### Step 1: Implementing the Forward Pass

First, we define the function that performs the actual computation. For the sake of simplicity we leave out vectorization, parallelization and broadcasting:

```python
# Import necessary modules
from endia.utils import op_array, setup_shape_and_data, clone_shape
from endia import Array

# Define the forward pass as a low-level function
def custom_mul_callable(inout curr: Array, args: List[Array]) -> None:
    setup_shape_and_data(curr)
    for i in range(curr.size()):
        curr.store(i, args[0].load(i) * args[1].load(i))
```

**This function:**

- Initializes the output array using `setup_shape_and_data`
- Performs elementwise multiplication of the input arrays
- Stores the result in the output array

### Step 2: Defining the Backward Pass

We implement a vector-Jacobian product (vjp) function for efficient reverse-mode automatic differentiation:

```python
def custom_mul_vjp(primals: List[Array], grad: Array, out: Array) -> List[Array]:
    return List(custom_mul(grad, primals[1]), custom_mul(grad, primals[0]))
```

***Mathematical Explanation:*** 

In the following, we denote: `f` as the custom_mul_vjp, `x` as primals[0], `y` as primals[1], `v` as the incoming gradient `grad` adn `L` as the scalar output of the entire differentiable program.


    - *Objective:* Propagate the incoming (given) gradient `v = ∂L/∂f` backwards through our custom function. 
      
    - *Derivative Computation:* To determine how small input changes affect the output of our custom function, we calculate its **Partial Derivatives**. For our multiplication function `f(x,y) = x * y`, we differentiate 
      with respect to each input variable `x` and `y`:
        `∂f/∂x = y` and `∂f/∂y = x`.

    - *Jacobian Formation:* We construct the **Jacobian Matrix** by arranging these partial derivatives:
        `J = [∂f/∂x, ∂f/∂y] = [y, x]`

    - *Gradient Propagation:* Applying the chain rule from calculus, we compute the **Vector-Jacobian Product** 
      to obtain the derivatives `∂L/∂x` and `∂L/∂y`:
        `[∂L/∂x, ∂L/y] = v * J = ∂L/∂f * [y, x] = [∂L/∂f * y, ∂L/∂f * x]`


***Note 1: The Mojo compiler***: 

The Mojo compiler is smart enough to have us first define the `vjp` function using the custom_mul, and then define the custom_mul itself.

***Note 2: On Differentiation in Endia:***

A key advantage of Endia's design is that we can define the backward pass using existing differentiable operations. This means we don't need to implement the low-level details of gradient computation ourselves. Instead, we can express the gradient calculation in terms of operations that are already differentiable.
This approach is crucial for enabling **higher-order differentiation** in Endia. When a function is purely defined in terms of differentiable smaller functions, we can differentiate the corresponding function as many times as we want. This is a powerful feature for various scientific computing applications, including optimization algorithms, differential equations solvers, and advanced machine learning models.

### Step 3: Registering the Operation

Finally, we register our custom operation with Endia:

```python
# Register the operation with Endia
def custom_mul(arg0: Array, arg1: Array) -> Array:
    return op_array(
        array_shape=clone_shape(arg0.array_shape()),
        args=List(arg0, arg1),
        name="mul",
        callable=custom_mul_callable,
        vjp=custom_mul_vjp
    )
```

**This function:**

- Uses the Endia utility function `op_array` to register the operation
- Specifies the output shape, input arrays, operation name, and callable functions

## Using the Custom Operation

Here's how to use our new custom operation:

```python
def main():
    a = Array('[1, 2, 3]', requires_grad=True)
    b = Array('[4, 5, 6]', requires_grad=True)

    c = endia.sum(custom_mul(a, b))
    print(c)  # Expected: [32]

    c.backward()
    print(a.grad())  # Expected: [4, 5, 6]
    print(b.grad())  # Expected: [1, 2, 3]
```

## That's it!

Congratulations! You've created a custom differentiable operation in Endia. This capability opens up a world of possibilities for extending Endia's functionality to meet your specific needs. Whether you're optimizing existing operations or developing entirely new ones, custom operations are a powerful tool for enhancing your scientific computing workflows. 🎉

## But there is more...

You can read through the [actual imeplementations the built-in mul operation](https://github.com/endia-ai/Endia/blob/main/endia/functional/binary_ops/mul_op.mojo) and you will see that they are implemented in a similar way, but with proper broadcasting capabilities, and a lot of optimizations for performance.

In the above guide we mainly focused on doing reverse-mode automatic differentiation. However, Endia will soon also support forward-mode automatic differentiation, where the notion of the vector-Jacobian product is replaced by the Jacobian-vector product. **_Stay tuned!_**
