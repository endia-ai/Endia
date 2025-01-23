import unittest
import torch
import torch.nn.functional as F

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
def transposed_convolution_1d_numba(input, kernel, stride=1, padding=0, output_padding=0, dilation=1, groups=1):
    """
    1D Transposed Convolution implemented with Numba.

    Args:
        input (np.ndarray): Input tensor of shape (batch_size, input_channels, input_w).
        kernel (np.ndarray): Kernel tensor of shape (input_channels, output_channels_per_group, kernel_w).
        stride (int, optional): Stride for the transposed convolution. Defaults to 1.
        padding (int, optional): Padding for the transposed convolution. Defaults to 0.
        output_padding (int, optional): Output padding for the transposed convolution. Defaults to 0.
        dilation (int, optional): Dilation for the transposed convolution. Defaults to 1.
        groups (int, optional): Number of groups for grouped convolution. Defaults to 1.

    Returns:
        np.ndarray: Output tensor of shape (batch_size, output_channels, output_w).
    """
    # Input validation
    batch_size, input_channels, input_w = input.shape
    in_channels_kernel, out_channels_kernel_per_group, kernel_w = kernel.shape

    output_channels = out_channels_kernel_per_group * groups

    # Validate groups
    assert input_channels % groups == 0, "Input channels must be divisible by groups"
    assert output_channels % groups == 0, "Output channels must be divisible by groups"
    assert in_channels_kernel == input_channels, "Kernel input channels should match input channels"

    # Handle tuple inputs for stride, padding, output_padding, dilation - now just int or tuple of length 1
    stride_w = stride if isinstance(stride, tuple) or isinstance(stride, list) else stride # Accept list for test compatibility
    stride_w = stride_w[0] if isinstance(stride_w, tuple) or isinstance(stride_w, list) else stride_w
    padding_w = padding if isinstance(padding, tuple) or isinstance(padding, list) else padding
    padding_w = padding_w[0] if isinstance(padding_w, tuple) or isinstance(padding_w, list) else padding_w
    output_padding_w = output_padding if isinstance(output_padding, tuple) or isinstance(output_padding, list) else output_padding
    output_padding_w = output_padding_w[0] if isinstance(output_padding_w, tuple) or isinstance(output_padding_w, list) else output_padding_w
    dilation_w = dilation if isinstance(dilation, tuple) or isinstance(dilation, list) else dilation
    dilation_w = dilation_w[0] if isinstance(dilation_w, tuple) or isinstance(dilation_w, list) else dilation_w


    # Calculate output dimensions
    effective_kernel_w = (kernel_w - 1) * dilation_w + 1
    output_w = (input_w - 1) * stride_w - 2 * padding_w + effective_kernel_w + output_padding_w

    # Initialize the output with zeros
    output = np.zeros((batch_size, output_channels, output_w), dtype=np.float32)

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
                for i_w in range(input_w):
                    # Calculate starting output position
                    start_w = i_w * stride_w - padding_w

                    # Iterate over kernel dimensions
                    for kw in range(kernel_w):
                        # Calculate output position
                        out_w = start_w + kw * dilation_w

                        # Check output bounds
                        if 0 <= out_w < output_w:
                            # Iterate over output channels in this group
                            for c_out_group in range(output_channels_per_group):
                                # Actual output channel index
                                c_out = output_group_start + c_out_group

                                # Perform convolution
                                output[b, c_out, out_w] += (
                                    input[b, c_in, i_w] *
                                    kernel[c_in, c_out_group, kw]
                                )

    return output

def compute_1d_gradients(batch_size, in_channels, length, out_channels, 
                        kernel_size, stride, padding, dilation, groups=1):
    """Compute both automatic and manual gradients for Conv1d operation.
    
    Args:
        batch_size (int): Number of samples in the batch
        in_channels (int): Number of input channels
        length (int): Length of input sequence
        out_channels (int): Number of output channels
        kernel_size (int): Size of the convolving kernel
        stride (int): Stride of the convolution
        padding (int): Padding added to both sides of the input
        dilation (int): Spacing between kernel elements
        groups (int): Number of blocked connections from input channels to output channels
    """
    # Validate inputs
    if kernel_size > length:
        raise ValueError(f"Kernel size ({kernel_size}) cannot be larger than input length ({length})")
    if any(x <= 0 for x in [batch_size, in_channels, length, out_channels, kernel_size, stride, dilation, groups]):
        raise ValueError("All input parameters except padding must be positive")
    if padding < 0:
        raise ValueError("Padding must be non-negative")
    if in_channels % groups != 0 or out_channels % groups != 0:
        raise ValueError(f"in_channels ({in_channels}) and out_channels ({out_channels}) must be divisible by groups ({groups})")

    # Initialize tensors
    torch.manual_seed(42)
    x = torch.randn(batch_size, in_channels, length, requires_grad=True)
    weight = torch.randn(out_channels, in_channels // groups, kernel_size, requires_grad=True)
    bias = torch.randn(out_channels, requires_grad=True)

    # Forward pass
    output = F.conv1d(x, weight, bias, stride=stride, padding=padding, 
                     dilation=dilation, groups=groups)
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
    output_padding = (length + 2 * padding - dilation * (kernel_size - 1) - 1) % stride
    manual_grad_x = transposed_convolution_1d_numba(
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


    # Weight gradient
    manual_grad_weight = torch.zeros_like(weight)
    for g in range(groups):
        x_g = x[:, g*(in_channels//groups):(g+1)*(in_channels//groups)]
        grad_output_g = grad_output[:, g*(out_channels//groups):(g+1)*(out_channels//groups)]
        
        x_unf = F.unfold(x_g.unsqueeze(2), 
                        kernel_size=(1, kernel_size),
                        dilation=(1, dilation),
                        padding=(0, padding),
                        stride=(1, stride))
        x_unf = x_unf.squeeze(2)
        
        grad_output_reshaped = grad_output_g.reshape(batch_size, out_channels//groups, -1)
        grad_weight_g = torch.bmm(grad_output_reshaped, x_unf.transpose(1, 2))
        grad_weight_g = grad_weight_g.sum(dim=0).reshape(out_channels//groups, in_channels//groups, kernel_size)
        
        manual_grad_weight[g*(out_channels//groups):(g+1)*(out_channels//groups)] = grad_weight_g

    # Bias gradient
    manual_grad_bias = grad_output.sum(dim=(0, 2))

    return {
        'auto_grad_x': auto_grad_x,
        'auto_grad_weight': auto_grad_weight,
        'auto_grad_bias': auto_grad_bias,
        'manual_grad_x': manual_grad_x,
        'manual_grad_weight': manual_grad_weight,
        'manual_grad_bias': manual_grad_bias
    }

class TestConv1dGradients(unittest.TestCase):
    def assertGradientsClose(self, grads, tolerance=1e-5):
        """Assert that automatic and manual gradients are close within tolerance."""
        for name in ['x', 'weight', 'bias']:
            auto_grad = grads[f'auto_grad_{name}']
            manual_grad = grads[f'manual_grad_{name}']
            
            self.assertGreater(torch.norm(auto_grad).item(), 1e-10)
            self.assertGreater(torch.norm(manual_grad).item(), 1e-10)
            
            relative_error = torch.norm(auto_grad - manual_grad) / torch.norm(auto_grad)
            self.assertLess(relative_error.item(), tolerance)

    def test_basic_case(self):
        """Test with basic configuration."""
        grads = compute_1d_gradients(
            batch_size=1, in_channels=1, length=16,
            out_channels=1, kernel_size=3, stride=1, padding=1, dilation=1
        )
        self.assertGradientsClose(grads)

    def test_channel_combinations(self):
        """Test different combinations of input and output channels."""
        configs = [(1, 1), (2, 4), (4, 2), (3, 3)]
        for in_ch, out_ch in configs:
            with self.subTest(in_channels=in_ch, out_channels=out_ch):
                grads = compute_1d_gradients(
                    batch_size=2, in_channels=in_ch, length=16,
                    out_channels=out_ch, kernel_size=3, stride=1, padding=1, dilation=1
                )
                self.assertGradientsClose(grads)

    def test_group_configurations(self):
        """Test different group configurations."""
        configs = [
            (4, 4, 2),  # Standard grouped conv
            (6, 6, 2),  # More channels
            (8, 8, 4),  # More groups
        ]
        for in_ch, out_ch, groups in configs:
            with self.subTest(in_ch=in_ch, out_ch=out_ch, groups=groups):
                grads = compute_1d_gradients(
                    batch_size=2, in_channels=in_ch, length=16,
                    out_channels=out_ch, kernel_size=3, stride=1, padding=1, 
                    dilation=1, groups=groups
                )
                self.assertGradientsClose(grads)

    def test_stride_variations(self):
        """Test different stride values."""
        for stride in [1, 2, 3]:
            with self.subTest(stride=stride):
                grads = compute_1d_gradients(
                    batch_size=2, in_channels=2, length=16,
                    out_channels=2, kernel_size=3, stride=stride, padding=1, dilation=1
                )
                self.assertGradientsClose(grads)

    def test_kernel_size_variations(self):
        """Test different kernel sizes."""
        for kernel_size in [1, 3, 5, 7]:
            with self.subTest(kernel_size=kernel_size):
                grads = compute_1d_gradients(
                    batch_size=2, in_channels=2, length=16,
                    out_channels=2, kernel_size=kernel_size, stride=1, 
                    padding=kernel_size//2, dilation=1
                )
                self.assertGradientsClose(grads)

    def test_dilation_variations(self):
        """Test different dilation values."""
        for dilation in [1, 2, 3]:
            with self.subTest(dilation=dilation):
                grads = compute_1d_gradients(
                    batch_size=2, in_channels=2, length=20,
                    out_channels=2, kernel_size=3, stride=1, padding=dilation, 
                    dilation=dilation
                )
                self.assertGradientsClose(grads)

    def test_invalid_configurations(self):
        """Test that invalid configurations raise appropriate exceptions."""
        invalid_configs = [
            # Kernel too large
            (2, 2, 4, 2, 5, 1, 1, 1),
            # Invalid groups
            (2, 3, 16, 4, 3, 1, 1, 1, 2),
            # Negative padding
            (2, 2, 16, 2, 3, 1, -1, 1),
        ]
        for config in invalid_configs:
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    compute_1d_gradients(*config)

if __name__ == '__main__':
    unittest.main(verbosity=2)
