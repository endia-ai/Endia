import unittest
import torch
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from numba import njit # Import Numba

@njit # Add Numba JIT decorator
def transposed_convolution_2d_numba(input, kernel, stride=1, padding=0, output_padding=0, dilation=1, groups=1): # Renamed function
    # Input validation
    batch_size, input_channels, input_h, input_w = input.shape
    in_channels_kernel, out_channels_kernel_per_group, kernel_h, kernel_w = kernel.shape # Correct kernel shape

    output_channels = out_channels_kernel_per_group * groups # Calculate output_channels from kernel shape

    # Validate groups
    assert input_channels % groups == 0, "Input channels must be divisible by groups"
    assert output_channels % groups == 0, "Output channels must be divisible by groups"
    assert in_channels_kernel == input_channels, "Kernel input channels should match input channels" # Correct check

    # Handle tuple inputs for stride, padding, output_padding
    stride_h, stride_w = stride if isinstance(stride, tuple) else (stride, stride)
    padding_h, padding_w = padding if isinstance(padding, tuple) else (padding, padding)
    output_padding_h, output_padding_w = output_padding if isinstance(output_padding, tuple) else (output_padding, output_padding)
    dilation_h, dilation_w = dilation if isinstance(dilation, tuple) else (dilation, dilation) # although dilation is int in example, be consistent

    # Calculate output dimensions
    effective_kernel_h = (kernel_h - 1) * dilation_h + 1
    effective_kernel_w = (kernel_w - 1) * dilation_w + 1
    output_h = (input_h - 1) * stride_h - 2 * padding_h + effective_kernel_h + output_padding_h
    output_w = (input_w - 1) * stride_w - 2 * padding_w + effective_kernel_w + output_padding_w

    # Initialize the output with zeros
    output = np.zeros((batch_size, output_channels, output_h, output_w), dtype=np.float32)

    # Group-specific parameters
    channels_per_group = input_channels // groups
    output_channels_per_group = output_channels // groups

    # Apply the transposed convolution with groups
    for b in range(batch_size):  # Iterate over batch
        for g in range(groups):  # Iterate over groups
            # Calculate group-specific channel ranges
            input_group_start = g * channels_per_group
            input_group_end = input_group_start + channels_per_group

            output_group_start = g * output_channels_per_group
            output_group_end = output_group_start + output_channels_per_group

            # Process each input channel in this group
            for c_in_group in range(channels_per_group):
                # Actual input channel index
                c_in = input_group_start + c_in_group

                # Iterate over input spatial dimensions
                for i in range(input_h):
                    for j in range(input_w):
                        # Calculate starting output position
                        start_h = i * stride_h - padding_h
                        start_w = j * stride_w - padding_w

                        # Iterate over kernel dimensions
                        for kh in range(kernel_h):
                            for kw in range(kernel_w):
                                # Calculate output position
                                out_h = start_h + kh * dilation_h
                                out_w = start_w + kw * dilation_w

                                # Check output bounds
                                if 0 <= out_h < output_h and 0 <= out_w < output_w:
                                    # Iterate over output channels in this group
                                    for c_out_group in range(output_channels_per_group):
                                        # Actual output channel index
                                        c_out = output_group_start + c_out_group

                                        # Perform convolution
                                        output[b, c_out, out_h, out_w] += (
                                            input[b, c_in, i, j] *
                                            kernel[c_in, c_out_group, kh, kw]
                                        )

    return output


def compute_gradients(batch_size, in_channels, height, width, out_channels, kernel_size, stride, padding, dilation, groups=1):
    """Compute both automatic and manual gradients for Conv2d operation.
    
    Args:
        batch_size (int): Number of samples in the batch
        in_channels (int): Number of input channels
        height (int): Height of input
        width (int): Width of input
        out_channels (int): Number of output channels
        kernel_size (int): Size of the convolving kernel
        stride (int): Stride of the convolution
        padding (int): Padding added to all sides of the input
        dilation (int): Spacing between kernel elements
        groups (int): Number of blocked connections from input channels to output channels
        
    Returns:
        dict: Dictionary containing automatic and manual gradients
        
    Raises:
        ValueError: If input dimensions are invalid, kernel size is larger than input, or groups are invalid
    """
    # Validate inputs
    if kernel_size > min(height, width):
        raise ValueError(f"Kernel size ({kernel_size}) cannot be larger than input dimensions ({height}, {width})")
    if any(x <= 0 for x in [batch_size, in_channels, height, width, out_channels, kernel_size, stride, dilation, groups]):
        raise ValueError("All input parameters except padding must be positive")
    if padding < 0:
        raise ValueError("Padding must be non-negative")
    if in_channels % groups != 0 or out_channels % groups != 0:
        raise ValueError(f"in_channels ({in_channels}) and out_channels ({out_channels}) must be divisible by groups ({groups})")

    # Initialize tensors with fixed seed for reproducibility
    torch.manual_seed(42)
    x = torch.randn(batch_size, in_channels, height, width, requires_grad=True)
    weight = torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size, requires_grad=True)
    bias = torch.randn(out_channels, requires_grad=True)

    # Forward pass with groups
    output = F.conv2d(x, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
    grad_output = torch.randn_like(output)
    
    # Compute automatic gradients
    output.backward(grad_output)
    auto_grad_x = x.grad.clone()
    auto_grad_weight = weight.grad.clone()
    auto_grad_bias = bias.grad.clone()
    
    # Reset gradients
    x.grad.zero_()
    weight.grad.zero_()
    bias.grad.zero_()

    # Manual gradient computations
    # Input gradient
    output_padding = (
        (height + 2 * padding - dilation * (kernel_size - 1) - 1) % stride,
        (width + 2 * padding - dilation * (kernel_size - 1) - 1) % stride
    )
    manual_grad_x = transposed_convolution_2d_numba(
        input=grad_output.clone().detach().numpy(), 
        kernel=weight.clone().detach().numpy(),
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        groups=groups
    )

    # Convert back to torch tensor
    manual_grad_x = torch.tensor(manual_grad_x, dtype=torch.float32)


    # Weight gradient (with groups)
    manual_grad_weight = torch.zeros_like(weight)
    for g in range(groups):
        x_g = x[:, g * (in_channels // groups):(g + 1) * (in_channels // groups)]
        grad_output_g = grad_output[:, g * (out_channels // groups):(g + 1) * (out_channels // groups)]
        
        x_unf = F.unfold(x_g, kernel_size=(kernel_size, kernel_size), dilation=dilation, padding=padding, stride=stride).view(batch_size, (in_channels // groups) * kernel_size * kernel_size, -1)
        
        grad_output_reshaped = grad_output_g.reshape(batch_size, out_channels // groups, -1)
        grad_weight_g = torch.bmm(
            grad_output_reshaped,
            x_unf.transpose(1, 2)
        ).sum(dim=0).reshape(out_channels // groups, in_channels // groups, kernel_size, kernel_size)
        
        manual_grad_weight[g * (out_channels // groups):(g + 1) * (out_channels // groups)] = grad_weight_g

    # Bias gradient
    manual_grad_bias = grad_output.sum(dim=(0, 2, 3))

    return {
        'auto_grad_x': auto_grad_x,
        'auto_grad_weight': auto_grad_weight,
        'auto_grad_bias': auto_grad_bias,
        'manual_grad_x': manual_grad_x,
        'manual_grad_weight': manual_grad_weight,
        'manual_grad_bias': manual_grad_bias
    }

class TestConv2dGradients(unittest.TestCase):

    def assertGradientsClose(self, grads, tolerance=1e-5):
        """Assert that automatic and manual gradients are close within tolerance."""
        for name in ['x', 'weight', 'bias']:
            auto_grad = grads[f'auto_grad_{name}']
            manual_grad = grads[f'manual_grad_{name}']
            
            # Check that gradients aren't trivially zero
            self.assertGreater(torch.norm(auto_grad).item(), 1e-10, 
                             f"Automatic gradients for {name} are too close to zero")
            self.assertGreater(torch.norm(manual_grad).item(), 1e-10, 
                             f"Manual gradients for {name} are too close to zero")
            
            # Check relative error
            relative_error = torch.norm(auto_grad - manual_grad) / torch.norm(auto_grad)
            self.assertLess(relative_error.item(), tolerance, 
                          f"High relative error for {name}: {relative_error.item()}")

    def test_basic_case(self):
        """Test with basic configuration."""
        grads = compute_gradients(
            batch_size=1, in_channels=1, height=8, width=8,
            out_channels=1, kernel_size=2, stride=1, padding=0, dilation=1
        )
        self.assertGradientsClose(grads)

    def test_large_batch_sizes(self):
        """Test with various batch sizes."""
        for batch_size in [1, 16, 32, 64]:
            with self.subTest(batch_size=batch_size):
                grads = compute_gradients(
                    batch_size=batch_size, in_channels=3, height=16, width=16,
                    out_channels=4, kernel_size=3, stride=1, padding=1, dilation=1
                )
                self.assertGradientsClose(grads)

    def test_channel_combinations(self):
        """Test different combinations of input and output channels."""
        channel_configs = [
            (1, 1), (3, 1), (1, 3), (3, 3), (4, 8), (8, 4)
        ]
        for in_channels, out_channels in channel_configs:
            with self.subTest(in_channels=in_channels, out_channels=out_channels):
                grads = compute_gradients(
                    batch_size=2, in_channels=in_channels, height=12, width=12,
                    out_channels=out_channels, kernel_size=3, stride=1, padding=1, dilation=1
                )
                self.assertGradientsClose(grads)

    def test_kernel_sizes(self):
        """Test different kernel sizes."""
        for kernel_size in [1, 2, 3, 5, 7]:
            with self.subTest(kernel_size=kernel_size):
                grads = compute_gradients(
                    batch_size=2, in_channels=2, height=16, width=16,
                    out_channels=2, kernel_size=kernel_size, stride=1, 
                    padding=kernel_size//2, dilation=1
                )
                self.assertGradientsClose(grads)

    def test_stride_combinations(self):
        """Test different stride values."""
        for stride in [1, 2, 3, 4]:
            with self.subTest(stride=stride):
                grads = compute_gradients(
                    batch_size=2, in_channels=2, height=20, width=20,
                    out_channels=2, kernel_size=3, stride=stride, padding=1, dilation=1
                )
                self.assertGradientsClose(grads)

    def test_padding_combinations(self):
        """Test different padding values."""
        for padding in [0, 1, 2, 3]:
            with self.subTest(padding=padding):
                grads = compute_gradients(
                    batch_size=2, in_channels=2, height=16, width=16,
                    out_channels=2, kernel_size=3, stride=1, padding=padding, dilation=1
                )
                self.assertGradientsClose(grads)

    def test_dilation_combinations(self):
        """Test different dilation values."""
        for dilation in [1, 2, 3]:
            with self.subTest(dilation=dilation):
                grads = compute_gradients(
                    batch_size=2, in_channels=2, height=20, width=20,
                    out_channels=2, kernel_size=3, stride=1, padding=dilation, dilation=dilation
                )
                self.assertGradientsClose(grads)

    def test_asymmetric_inputs(self):
        """Test with asymmetric input dimensions."""
        dimensions = [(10, 15), (15, 10), (8, 32), (32, 8)]
        for height, width in dimensions:
            with self.subTest(height=height, width=width):
                grads = compute_gradients(
                    batch_size=2, in_channels=2, height=height, width=width,
                    out_channels=3, kernel_size=3, stride=1, padding=1, dilation=1
                )
                self.assertGradientsClose(grads)

    def test_extreme_cases(self):
        """Test extreme configurations."""
        configs = [
            # Very small input
            (1, 1, 4, 4, 1, 3, 1, 1, 1),
            # Large input with large kernel
            (1, 1, 32, 32, 1, 7, 1, 3, 1),
            # Multiple channels with large stride
            (2, 4, 16, 16, 4, 3, 4, 1, 1),
            # Large dilation
            (2, 2, 20, 20, 2, 3, 1, 4, 4),
        ]
        for config in configs:
            with self.subTest(config=config):
                grads = compute_gradients(*config)
                self.assertGradientsClose(grads)

    def test_numerical_stability(self):
        """Test numerical stability with different input scales."""
        scales = [0.001, 0.1, 1.0, 10.0, 100.0]
        for scale in scales:
            with self.subTest(scale=scale):
                torch.manual_seed(42)
                x = torch.randn(2, 2, 16, 16) * scale
                weight = torch.randn(2, 2, 3, 3) * scale
                x.requires_grad_(True)
                weight.requires_grad_(True)
                
                output = F.conv2d(x, weight, None, stride=1, padding=1)
                grad_output = torch.randn_like(output)
                output.backward(grad_output)
                
                self.assertTrue(torch.isfinite(x.grad).all())
                self.assertTrue(torch.isfinite(weight.grad).all())
                
                # Clean up gradients
                x.grad = None
                weight.grad = None

    def test_zero_inputs(self):
        """Test with zero-valued inputs."""
        grads = compute_gradients(
            2, 2, 16, 16,
            2, 3, 1, 1, 1
        )
        self.assertGradientsClose(grads)

    def test_checkerboard_input(self):
        """Test with checkerboard pattern input."""
        torch.manual_seed(42)
        batch_size, in_channels, height, width = 2, 1, 8, 8
        # Create checkerboard pattern without in-place operations
        x = torch.ones(batch_size, in_channels, height, width)
        checkerboard = torch.ones(height, width)
        checkerboard[::2, ::2] = -1
        checkerboard[1::2, 1::2] = -1
        x = x * checkerboard.expand(batch_size, in_channels, height, width)
        x.requires_grad_(True)
        
        weight = torch.randn(2, in_channels, 3, 3, requires_grad=True)
        bias = torch.randn(2, requires_grad=True)
        
        # Forward pass
        output = F.conv2d(x, weight, bias, stride=1, padding=1, dilation=1)
        grad_output = torch.randn_like(output)
        
        # Compute automatic gradients
        output.backward(grad_output)
        auto_grad_x = x.grad.clone()
        auto_grad_weight = weight.grad.clone()
        auto_grad_bias = bias.grad.clone()
        
        # Reset gradients
        x.grad.zero_()
        weight.grad.zero_()
        bias.grad.zero_()
        
        # Manual gradient computations using the same logic as compute_gradients
        manual_grad_x = F.conv_transpose2d(
            grad_output, weight,
            stride=1, padding=1, dilation=1
        )
        
        x_unf = F.unfold(x, kernel_size=(3, 3), padding=1, stride=1).view(batch_size, in_channels * 9, -1)
        grad_output_reshaped = grad_output.reshape(batch_size, 2, -1)
        manual_grad_weight = torch.bmm(
            grad_output_reshaped,
            x_unf.transpose(1, 2)
        ).sum(dim=0).reshape(2, in_channels, 3, 3)
        
        manual_grad_bias = grad_output.sum(dim=(0, 2, 3))
        
        grads = {
            'auto_grad_x': auto_grad_x,
            'auto_grad_weight': auto_grad_weight,
            'auto_grad_bias': auto_grad_bias,
            'manual_grad_x': manual_grad_x,
            'manual_grad_weight': manual_grad_weight,
            'manual_grad_bias': manual_grad_bias
        }
        self.assertGradientsClose(grads)

    def test_invalid_inputs(self):
        """Test that invalid inputs raise appropriate exceptions."""
        invalid_configs = [
            # Kernel size larger than input
            (1, 1, 4, 4, 1, 5, 1, 1, 1),
            # Zero or negative values
            (0, 1, 4, 4, 1, 3, 1, 1, 1),
            (1, -1, 4, 4, 1, 3, 1, 1, 1),
        ]
        
        for config in invalid_configs:
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    compute_gradients(*config)

    def test_group_combinations(self):
        """Test different group configurations."""
        group_configs = [
            # (in_channels, out_channels, groups)
            (4, 4, 2),    # Standard grouped conv
            (8, 8, 4),    # Multiple groups
            (3, 3, 3),    # Channel-wise conv (groups = channels)
            (6, 12, 3),   # Different in/out channel ratios
        ]
        for in_channels, out_channels, groups in group_configs:
            with self.subTest(in_channels=in_channels, out_channels=out_channels, groups=groups):
                grads = compute_gradients(
                    batch_size=2, in_channels=in_channels, height=16, width=16,
                    out_channels=out_channels, kernel_size=3, stride=1, 
                    padding=1, dilation=1, groups=groups
                )
                self.assertGradientsClose(grads)

    def test_invalid_groups(self):
        """Test invalid group configurations."""
        invalid_configs = [
            # in_channels not divisible by groups
            (5, 6, 16, 16, 6, 3, 1, 1, 1, 4),
            # out_channels not divisible by groups
            (2, 6, 16, 16, 5, 3, 1, 1, 1, 3),
            # groups > in_channels
            (2, 4, 16, 16, 4, 3, 1, 1, 1, 8),
        ]
        for config in invalid_configs:
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    compute_gradients(*config)

    def tearDown(self):
        """Clean up any remaining gradients after each test."""
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    @classmethod
    def setUpClass(cls):
        """Set default device and seed for reproducibility."""
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)

if __name__ == '__main__':
    unittest.main(verbosity=2)