import endia as nd

fn main() raises:
    var a = nd.array("[[1, 2, 3], [4, 5, 6]]", requires_grad=True)
    print(a)

    var a_flipped = nd.flip(a, List(-1, -2))
    print(a_flipped)

    a_flipped.grad_(nd.array("[[1, 2, 3], [4, 5, 6]]"))
    var grads = nd.grad(outs=List(a_flipped), inputs=List(a))
    print(grads[0])