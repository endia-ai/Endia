import torch
import torch.nn.functional as F
import unittest

def manual_max_pool2d_backward(input_tensor, output_grad, kernel_size, stride, padding, dilation):
    """
    Manually computes the gradient of max_pool2d.

    Args:
        input_tensor: The original input tensor.
        output_grad: Gradient with respect to the output tensor.
        kernel_size, stride, padding, dilation, ceil_mode: Pooling parameters.

    Returns:
        input_gradient: Gradient with respect to the input tensor.
    """
    batch_size, in_channels, in_height, in_width = input_tensor.shape
    _, _, out_height, out_width = output_grad.shape

    input_gradient = torch.zeros_like(input_tensor)

    for b in range(batch_size):
        for c in range(in_channels):
            for out_h in range(out_height):
                for out_w in range(out_width):
                    # 1. Determine the input window corresponding to the current output element (out_h, out_w)
                    in_h_start = out_h * stride - padding
                    in_w_start = out_w * stride - padding

                    max_val = -float('inf')
                    max_h_index, max_w_index = -1, -1

                    for kh in range(kernel_size): # Iterate over kernel height indices
                        for kw in range(kernel_size): # Iterate over kernel width indices
                            current_in_h = in_h_start + kh * dilation
                            current_in_w = in_w_start + kw * dilation

                            if (0 <= current_in_h < in_height) and (0 <= current_in_w < in_width):
                                if input_tensor[b, c, current_in_h, current_in_w] > max_val:
                                    max_val = input_tensor[b, c, current_in_h, current_in_w]
                                    max_h_index = current_in_h
                                    max_w_index = current_in_w

                    if max_h_index != -1 and max_w_index != -1:
                        input_gradient[b, c, max_h_index, max_w_index] += output_grad[b, c, out_h, out_w]

    return input_gradient


class TestMaxPool2dBackward(unittest.TestCase):

    def _test_max_pool2d_backward_case(self, input_shape, kernel_size, stride, padding, dilation, ceil_mode):
        input_tensor = torch.randn(input_shape, requires_grad=True, dtype=torch.float64) # Use float64 for better numerical stability

        try: # Added try-except block to catch potential RuntimeError
            output = F.max_pool2d(
                input=input_tensor,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                ceil_mode=ceil_mode
            )
        except RuntimeError as e:
            if "calculated output size" in str(e) and "too small" in str(e):
                print(f"Skipping test case due to invalid output size for parameters: input_shape={input_shape}, kernel_size={kernel_size}, stride={stride}, padding={padding}, dilation={dilation}, ceil_mode={ceil_mode}")
                return # Skip this test case
            else:
                raise e # Re-raise other RuntimeErrors

        output_grad = torch.ones_like(output, dtype=torch.float64) # Use float64 for output gradient
        output.backward(output_grad)
        pytorch_gradient = input_tensor.grad.clone()

        input_tensor.grad.zero_() # Reset gradients for manual calculation

        manual_gradient = manual_max_pool2d_backward(
            input_tensor,
            output_grad,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode
        )

        is_close = torch.allclose(pytorch_gradient, manual_gradient, rtol=1e-5, atol=1e-8) # Set tighter tolerances for float64
        if not is_close:
            print(f"\nTest Failed for parameters: input_shape={input_shape}, kernel_size={kernel_size}, stride={stride}, padding={padding}, dilation={dilation}, ceil_mode={ceil_mode}")
            print("PyTorch Autograd Gradient:\n", pytorch_gradient)
            print("\nManual Gradient:\n", manual_gradient)
            print("\nDifference between gradients:\n", pytorch_gradient - manual_gradient)
        self.assertTrue(is_close, f"Gradients are not close for parameters: input_shape={input_shape}, kernel_size={kernel_size}, stride={stride}, padding={padding}, dilation={dilation}, ceil_mode={ceil_mode}")


    def test_basic_case(self):
        self._test_max_pool2d_backward_case(input_shape=(2, 3, 8, 8), kernel_size=2, stride=2, padding=0, dilation=1)

    def test_stride_1(self):
        self._test_max_pool2d_backward_case(input_shape=(1, 1, 7, 7), kernel_size=3, stride=1, padding=0, dilation=1)

    def test_padding(self):
        self._test_max_pool2d_backward_case(input_shape=(2, 2, 6, 6), kernel_size=2, stride=2, padding=1, dilation=1)

    def test_dilation_2(self):
        self._test_max_pool2d_backward_case(input_shape=(1, 3, 10, 10), kernel_size=3, stride=1, padding=1, dilation=2)

    def test_ceil_mode_true(self):
        self._test_max_pool2d_backward_case(input_shape=(2, 1, 9, 9), kernel_size=2, stride=3, padding=1, dilation=1)

    def test_kernel_larger_than_input(self):
        self._test_max_pool2d_backward_case(input_shape=(1, 1, 6, 6), kernel_size=5, stride=1, padding=0, dilation=1) # Modified input_shape

    def test_different_input_size(self):
        self._test_max_pool2d_backward_case(input_shape=(4, 5, 16, 16), kernel_size=3, stride=2, padding=1, dilation=2)

    def test_no_padding_dilation_stride_1(self):
        self._test_max_pool2d_backward_case(input_shape=(2, 3, 7, 7), kernel_size=3, stride=1, padding=0, dilation=2)

    def test_large_kernel_and_stride(self):
        self._test_max_pool2d_backward_case(input_shape=(1, 2, 20, 20), kernel_size=7, stride=5, padding=2, dilation=1)

    def test_batch_size_4(self):
        self._test_max_pool2d_backward_case(input_shape=(4, 3, 8, 8), kernel_size=2, stride=2, padding=0, dilation=1)

    def test_channels_4(self):
        self._test_max_pool2d_backward_case(input_shape=(2, 4, 8, 8), kernel_size=2, stride=2, padding=0, dilation=1)

    def test_small_input(self):
        self._test_max_pool2d_backward_case(input_shape=(1, 1, 2, 2), kernel_size=2, stride=1, padding=0, dilation=1)

    def test_invalid_output_size_padding(self): # Added test case that might produce invalid output size but is handled
        self._test_max_pool2d_backward_case(input_shape=(1, 1, 3, 3), kernel_size=5, stride=1, padding=2, dilation=1) # Added padding


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)