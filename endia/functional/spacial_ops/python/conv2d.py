import numpy as np
import time
import torch
import torch.nn.functional as F
from numpy.lib.stride_tricks import as_strided

def naive_conv2d(input_image, kernel, stride=1, padding=0, dilation=1, groups=1):
    """
    Naive implementation of 2D convolution with dilation and groups.
    """
    input_height, input_width, in_channels = input_image.shape
    kernel_height, kernel_width, kernel_in_channels, out_channels = kernel.shape

    assert in_channels % groups == 0, "Input channels must be divisible by groups"
    assert out_channels % groups == 0, "Output channels must be divisible by groups"
    assert kernel_in_channels == in_channels // groups, "Kernel in_channels must match input channels / groups"

    # Padding
    if padding > 0:
        input_padded = np.pad(input_image, ((padding, padding), (padding, padding), (0, 0)), mode='constant')
    else:
        input_padded = input_image
    padded_height, padded_width, _ = input_padded.shape

    output_height = (padded_height - dilation * (kernel_height - 1) - 1) // stride + 1
    output_width = (padded_width - dilation * (kernel_width - 1) - 1) // stride + 1
    output_feature_map = np.zeros((output_height, output_width, out_channels), dtype=np.float32)

    for g in range(groups):
        group_in_channels = in_channels // groups
        group_out_channels = out_channels // groups
        input_group = input_padded[:, :, g * group_in_channels:(g + 1) * group_in_channels]
        kernel_group = kernel[:, :, :, g * group_out_channels:(g + 1) * group_out_channels]

        for och in range(group_out_channels):
            for oh in range(output_height):
                for ow in range(output_width):
                    # Extract dilated input patch
                    input_patch = input_group[oh * stride:oh * stride + dilation * kernel_height:dilation,
                                               ow * stride:ow * stride + dilation * kernel_width:dilation,
                                               :]

                    # Handle cases where dilated patch is smaller than kernel due to image boundary
                    valid_kernel_height = input_patch.shape[0]
                    valid_kernel_width = input_patch.shape[1]
                    valid_kernel = kernel_group[:valid_kernel_height, :valid_kernel_width, :, och]

                    # Perform convolution (element-wise multiplication and sum)
                    output_feature_map[oh, ow, g * group_out_channels + och] = np.sum(input_patch * valid_kernel)

    return output_feature_map


def im2col(input_image, kernel_height, kernel_width, stride, padding, dilation, groups):
    """
    Transforms the input image into columns (im2col) with dilation and groups.
    """
    input_height, input_width, in_channels = input_image.shape

    if padding > 0:
        input_padded = np.pad(input_image, ((padding, padding), (padding, padding), (0, 0)), mode='constant')
    else:
        input_padded = input_image
    padded_height, padded_width, _ = input_padded.shape

    output_height = (padded_height - dilation * (kernel_height - 1) - 1) // stride + 1
    output_width = (padded_width - dilation * (kernel_width - 1) - 1) // stride + 1

    col_matrix = np.zeros((kernel_height * kernel_width * (in_channels // groups), output_height * output_width * groups), dtype=np.float32)

    for g in range(groups):
        col_idx = 0 # Moved col_idx initialization inside the group loop
        group_in_channels = in_channels // groups
        input_group = input_padded[:, :, g * group_in_channels:(g + 1) * group_in_channels]

        for oh in range(output_height):
            for ow in range(output_width):
                # Extract dilated patch and flatten it into a column
                patch = input_group[oh * stride:oh * stride + dilation * kernel_height:dilation,
                                     ow * stride:ow * stride + dilation * kernel_width:dilation,
                                     :].reshape(-1)  # Flatten to a column
                col_matrix[:, col_idx + g * output_height * output_width] = patch # Adjusted column index for groups
                col_idx += 1

    return col_matrix


def im2col_conv2d(input_image, kernel, stride=1, padding=0, dilation=1, groups=1):
    """
    Conv2D implementation using im2col and matrix multiplication with dilation and groups.
    """
    kernel_height, kernel_width, in_channels, out_channels = kernel.shape
    kernel_in_channels = kernel.shape[2] # Correct kernel_in_channels

    # im2col transformation
    col_matrix = im2col(input_image, kernel_height, kernel_width, stride, padding, dilation, groups)

    # Reshape kernel to matrix (out_channels, kernel_height * kernel_width * kernel_in_channels)
    kernel_matrix = kernel.reshape(kernel_height * kernel_width * kernel_in_channels, out_channels).T # Correct reshape


    # Matrix multiplication
    output_col_matrix = np.dot(kernel_matrix, col_matrix)

    # Reshape output to feature map
    if padding > 0:
        padded_height, padded_width, _ = np.pad(input_image, ((padding, padding), (padding, padding), (0, 0)), mode='constant').shape
    else:
        padded_height, padded_width, _ = input_image.shape

    output_height = (padded_height - dilation * (kernel_height - 1) - 1) // stride + 1
    output_width = (padded_width - dilation * (kernel_width - 1) - 1) // stride + 1
    output_feature_map = output_col_matrix.reshape(out_channels, output_height, output_width).transpose(1, 2, 0)

    return output_feature_map

def im2patch(input_image, kernel_height, kernel_width, stride, padding, dilation, groups):
    """
    Transforms the input image into patches (im2patch) with dilation and groups. (Original implementation)
    """
    input_height, input_width, in_channels = input_image.shape

    if padding > 0:
        input_padded = np.pad(input_image, ((padding, padding), (padding, padding), (0, 0)), mode='constant')
    else:
        input_padded = input_image
    padded_height, padded_width, _ = input_padded.shape

    output_height = (padded_height - dilation * (kernel_height - 1) - 1) // stride + 1
    output_width = (padded_width - dilation * (kernel_width - 1) - 1) // stride + 1

    patch_matrix = np.zeros((output_height * output_width * groups, kernel_height, kernel_width, (in_channels // groups)), dtype=np.float32) # Adjusted patch_matrix size

    for g in range(groups):
        patch_idx = 0 # Moved patch_idx initialization inside the group loop
        group_in_channels = in_channels // groups
        input_group = input_padded[:, :, g * group_in_channels:(g + 1) * group_in_channels]
        for oh in range(output_height):
            for ow in range(output_width):
                # Extract dilated patch
                patch = input_group[oh * stride:oh * stride + dilation * kernel_height:dilation,
                                     ow * stride:ow * stride + dilation * kernel_width:dilation,
                                     :]
                patch_matrix[patch_idx + g * output_height * output_width, :, :, :] = patch # Adjusted patch index for groups
                patch_idx += 1

    return patch_matrix


def im2patch_strided(input_image, kernel_height, kernel_width, stride, padding, dilation, groups):
    """
    Optimized im2patch using as_strided for faster patch extraction.
    """
    input_height, input_width, in_channels = input_image.shape

    if padding > 0:
        input_padded = np.pad(input_image, ((padding, padding), (padding, padding), (0, 0)), mode='constant')
    else:
        input_padded = input_image
    padded_height, padded_width, _ = input_padded.shape

    output_height = (padded_height - dilation * (kernel_height - 1) - 1) // stride + 1
    output_width = (padded_width - dilation * (kernel_width - 1) - 1) // stride + 1

    patch_shape = (kernel_height, kernel_width, in_channels // groups)
    patch_strides = (input_padded.strides[0] * dilation, input_padded.strides[1] * dilation, input_padded.strides[2]) # Strides for dilated kernel

    patch_matrix_list = []
    for g in range(groups):
        input_group = input_padded[:, :, g * (in_channels // groups):(g + 1) * (in_channels // groups)]
        patches_g = as_strided(
            input_group,
            shape=(output_height, output_width, *patch_shape), # (OH, OW, KH, KW, C_in/groups)
            strides=(stride * input_padded.strides[0], stride * input_padded.strides[1], *patch_strides) # Strides for output and kernel dims
        )
        patch_matrix_list.append(patches_g.reshape(output_height * output_width, *patch_shape)) # Flatten OH*OW dims

    patch_matrix = np.concatenate(patch_matrix_list, axis=0) # Concatenate along patch index

    return patch_matrix



def im2patch_conv2d(input_image, kernel, stride=1, padding=0, dilation=1, groups=1, use_strided_im2patch=False): # Added use_strided_im2patch flag
    """
    Conv2D implementation using im2patch and matrix multiplication with dilation and groups.
    """
    kernel_height, kernel_width, in_channels, out_channels = kernel.shape
    kernel_in_channels = kernel.shape[2] # Correct kernel_in_channels

    # im2patch transformation
    if use_strided_im2patch: # Choose between original and strided im2patch
        patch_matrix = im2patch_strided(input_image, kernel_height, kernel_width, stride, padding, dilation, groups)
    else:
        patch_matrix = im2patch(input_image, kernel_height, kernel_width, stride, padding, dilation, groups)


    # Reshape kernel matrix (out_channels, kernel_height * kernel_width * kernel_in_channels)
    kernel_matrix = kernel.reshape(kernel_height * kernel_width * kernel_in_channels, out_channels) # Correct reshape


    # Reshape patches to be compatible for matrix multiplication (output_height * output_width * groups, kernel_height * kernel_width * (in_channels // groups))
    patch_matrix_reshaped = patch_matrix.reshape(patch_matrix.shape[0], -1)

    # Matrix multiplication
    output_col_matrix = np.dot(patch_matrix_reshaped, kernel_matrix)

    # Reshape output to feature map
    if padding > 0:
        padded_height, padded_width, _ = np.pad(input_image, ((padding, padding), (padding, padding), (0, 0)), mode='constant').shape
    else:
        padded_height, padded_width, _ = input_image.shape

    output_height = (padded_height - dilation * (kernel_height - 1) - 1) // stride + 1
    output_width = (padded_width - dilation * (kernel_width - 1) - 1) // stride + 1
    output_feature_map = output_col_matrix.reshape(output_height * output_width * groups, out_channels // groups) # Reshape per group
    output_feature_map = output_feature_map.reshape(output_height, output_width, out_channels) # Combine groups to final output

    return output_feature_map

def pytorch_conv2d(input_image, kernel, stride=1, padding=0, dilation=1, groups=1):
    """
    Conv2D implementation using PyTorch for verification with dilation and groups.
    """
    # PyTorch expects input and kernel in a different format: (N, C, H, W) and (out_channels, in_channels, kernel_height, kernel_width)
    input_torch = torch.from_numpy(input_image).permute(2, 0, 1).unsqueeze(0).float()
    kernel_torch = torch.from_numpy(kernel).permute(3, 2, 0, 1).float()

    output_torch = F.conv2d(input_torch, kernel_torch, stride=stride, padding=padding, dilation=dilation, groups=groups)

    # Convert back to numpy and original format (H, W, C)
    output_numpy = output_torch.squeeze(0).permute(1, 2, 0).numpy()
    return output_numpy


def test_conv2d_implementations(input_shape, kernel_shape, strides, paddings, dilations, groups_list):
    """
    Rigorous testing function for Conv2D implementations.
    """
    input_image = np.random.rand(*input_shape).astype(np.float32)
    kernel = np.random.rand(*kernel_shape).astype(np.float32)

    test_results = {}

    for stride in strides:
        for padding in paddings:
            for dilation in dilations:
                for groups in groups_list:
                    if input_shape[2] % groups != 0 or kernel_shape[3] % groups != 0 or kernel_shape[2] // groups != kernel_shape[2]: # Correct group check
                        continue # Skip invalid group configurations

                    params_str = f"stride={stride}, padding={padding}, dilation={dilation}, groups={groups}"
                    # print(f"Testing with parameters: {params_str}")

                    # --- PyTorch Conv2D (for verification) ---
                    output_pytorch = pytorch_conv2d(input_image.copy(), kernel.copy(), stride, padding, dilation, groups)

                    # --- im2col Conv2D ---
                    output_im2col = im2col_conv2d(input_image.copy(), kernel.copy(), stride, padding, dilation, groups)

                    # --- im2patch Conv2D (im2win) ---
                    output_im2patch = im2patch_conv2d(input_image.copy(), kernel.copy(), stride, padding, dilation, groups, use_strided_im2patch=True)

                    # --- Verification ---
                    results = {}
                    # results['Naive vs PyTorch'] = np.allclose(output_pytorch, output_naive, rtol=1e-5, atol=1e-8)
                    results['im2col vs PyTorch'] = np.allclose(output_pytorch, output_im2col, rtol=1e-5, atol=1e-8)
                    results['im2patch vs PyTorch'] = np.allclose(output_pytorch, output_im2patch, rtol=1e-5, atol=1e-8)
                    test_results[params_str] = results

                    if not all(results.values()):
                        raise (f"Tests failed : input_shape={input_shape}, kernel_shape={kernel_shape}, stride={strides}, padding={paddings}, dilation={dilations}, groups={groups}")

    # print("\nAll tests completed successfully!")
    return True

# main funciton for rigorous testing differnt parameters
if __name__ == '__main__':
    input_shapes = [(32, 32, 16), (64, 64, 32), (128, 128, 64)]  # Different input sizes to test
    kernel_shape_for_testing = (3, 3, 4, 32) # Keep this fixed kernel shape (kernel size and out_channels)
    strides = [1, 2]
    paddings = [0, 1, 2]
    dilations = [1, 2] # Reduced dilations for faster testing, can add more if needed
    groups_list = [1, 2, 4]  # Reduced groups for faster testing, can add more if needed
    test_result = True # Initialize to True

    for input_shape in input_shapes: # Loop through different input shapes
        for groups in groups_list:
            for stride in strides:
                for padding in paddings:
                    for dilation in dilations:
                        kernel_in_channels_per_group = input_shape[2] // groups # Calculate kernel in_channels based on CURRENT input_shape
                        kernel_shape = (kernel_shape_for_testing[0], kernel_shape_for_testing[1], kernel_in_channels_per_group, kernel_shape_for_testing[3]) # Use fixed kernel size and out_channels
                        print(f"\n--- Testing with input_shape={input_shape}, groups = {groups}, stride={stride}, padding={padding}, dilation={dilation}, kernel_shape = {kernel_shape} ---")
                        group_test_result = test_conv2d_implementations(input_shape, kernel_shape, [stride], [padding], [dilation], [groups]) # Pass single values as lists
                        if not group_test_result:
                            test_result = False # Set overall test result to False if any group test fails
                            print(f"Tests failed for this parameter combination")
                            break # Stop testing this group if one test fails
                    if not test_result: break # Break dilation loop if test failed
                if not test_result: break # Break padding loop if test failed
            if not test_result: break # Break stride loop if test failed
        if not test_result: break # Break groups loop if test failed

    else: # else block of for loop, executes if loop completes without break
        print("\nAll tests completed successfully!")

    input_shape = (32, 32, 16)  # Increased in_channels for groups testing
    kernel_shape_for_testing = (3, 3, 4, 32) # Keep this for testing loop
    strides = [1, 2]
    paddings = [0, 1, 2, 3]
    dilations = [1, 2, 3]
    groups_list = [1, 2, 4, 8, 16]  # Test various group sizes
    num_timing_runs = 10 # Number of timing runs to average over

    if test_result: # Only run time comparison if tests passed
        print(f"\nTime Comparison (Example Run - Timing may vary, Averaging over {num_timing_runs} runs):")
        input_image_timing = np.random.rand(*input_shape).astype(np.float32)
        # Define kernel_shape for time comparison to match input_shape and groups=1
        kernel_shape_timing = (3, 3, input_shape[2], 32) # Use input_shape[2] for kernel in_channels
        kernel_timing = np.random.rand(*kernel_shape_timing).astype(np.float32)
        stride_timing = 1
        padding_timing = 1
        dilation_timing = 1
        groups_timing = 1

        pytorch_times = []
        naive_times = []
        im2col_times = []
        im2patch_times = [] # Changed to im2patch_times, now using strided im2patch as default

        for _ in range(num_timing_runs): # Loop for time averaging
            # --- PyTorch Conv2D ---
            start_time = time.time()
            pytorch_conv2d(input_image_timing.copy(), kernel_timing.copy(), stride_timing, padding_timing, dilation_timing, groups_timing)
            pytorch_times.append(time.time() - start_time)

            # --- im2col Conv2D ---
            start_time = time.time()
            im2col_conv2d(input_image_timing.copy(), kernel_timing.copy(), stride_timing, padding_timing, dilation_timing, groups_timing)
            im2col_times.append(time.time() - start_time)

            # --- im2patch Conv2d (im2win) - Strided (using strided im2patch as default now) ---
            start_time = time.time()
            im2patch_conv2d(input_image_timing.copy(), kernel_timing.copy(), stride_timing, padding_timing, dilation_timing, groups_timing, use_strided_im2patch=True) # Strided im2patch - using as default im2patch now
            im2patch_times.append(time.time() - start_time) # Changed to im2patch_times


        pytorch_time = np.mean(pytorch_times) # Average times
        # naive_time = np.mean(naive_times)
        im2col_time = np.mean(im2col_times)
        im2patch_time = np.mean(im2patch_times) # Changed to im2patch_time, now using strided im2patch as default


        # print(f"Naive Conv2D Time: {naive_time:.4f} seconds")
        print(f"im2col Conv2D Time: {im2col_time:.4f} seconds")
        print(f"im2patch Conv2D Time: {im2patch_time:.4f} seconds") # Changed to im2patch_time, now using strided im2patch as default
        print(f"PyTorch Conv2D Time: {pytorch_time:.4f} seconds") # Added avg runs info to output

        print("\nTime Comparison (Speedup - How many times faster than PyTorch, averaged over {num_timing_runs} runs):") # Added averaged runs info to speedup output
        # print(f"PyTorch / Naive: {naive_time / pytorch_time:.2f}x")
        print(f"PyTorch / im2col: {im2col_time / pytorch_time:.2f}x")
        print(f"PyTorch / im2patch: {im2patch_time / pytorch_time:.2f}x") # Changed to im2patch_time, now using strided im2patch as default


# main function for rigorous speed testing
# if __name__ == '__main__':
#     input_shapes = [(32, 32, 16), (64, 64, 16), (128, 128, 16)] # Different input sizes
#     kernel_sizes = [(3, 3, 4, 32), (5, 5, 4, 32), (7, 7, 4, 32)] # Different kernel sizes
#     strides = [1] # Fixed stride for time comparison
#     paddings = [1] # Fixed padding for time comparison
#     dilations = [1] # Fixed dilation for time comparison
#     groups_list = [1] # Fixed groups for time comparison
#     num_timing_runs = 10 # Number of timing runs to average over


#     for input_shape in input_shapes:
#         for kernel_shape_val in kernel_sizes: # Use kernel_sizes list
#             kernel_shape = kernel_shape_val # Assign current kernel shape

#             test_result = True # Reset test_result for each input/kernel size
#             for groups in groups_list: # Keep groups_list loop for potential future group testing
#                 kernel_in_channels_per_group = input_shape[2] // groups
#                 current_kernel_shape = (kernel_shape[0], kernel_shape[1], kernel_in_channels_per_group, kernel_shape[3]) # Use current kernel shape from kernel_sizes
#                 group_test_result = test_conv2d_implementations(input_shape, current_kernel_shape, strides, paddings, dilations, [groups])
#                 if not group_test_result:
#                     test_result = False # Set overall test result to False if any group test fails
#                     raise (f"Tests failed : input_shape={input_shape}, kernel_shape={current_kernel_shape}, stride={strides}, padding={paddings}, dilation={dilations}, groups={groups}")

#             if test_result: # Only run time comparison if tests passed
#                 print(f"\nTime Comparison (Input Shape: {input_shape}, Kernel Size: {(kernel_shape[0], kernel_shape[1])}, Averaging over {num_timing_runs} runs):") # Added input and kernel size info
#                 input_image_timing = np.random.rand(*input_shape).astype(np.float32)
#                 # Define kernel_shape for time comparison to match input_shape and groups=1
#                 kernel_shape_timing = (kernel_shape[0], kernel_shape[1], input_shape[2], kernel_shape[3]) # Use current kernel shape for timing
#                 kernel_timing = np.random.rand(*kernel_shape_timing).astype(np.float32)
#                 stride_timing = strides[0] # Use fixed stride
#                 padding_timing = paddings[0] # Use fixed padding
#                 dilation_timing = dilations[0] # Use fixed dilation
#                 groups_timing = groups_list[0] # Use fixed groups

#                 pytorch_times = []
#                 naive_times = []
#                 im2col_times = []
#                 im2patch_times = [] # Changed to im2patch_times, now using strided im2patch as default

#                 for _ in range(num_timing_runs): # Loop for time averaging
#                     # --- PyTorch Conv2D ---
#                     start_time = time.time()
#                     pytorch_conv2d(input_image_timing.copy(), kernel_timing.copy(), stride_timing, padding_timing, dilation_timing, groups_timing)
#                     pytorch_times.append(time.time() - start_time)

#                     # --- im2col Conv2D ---
#                     start_time = time.time()
#                     im2col_conv2d(input_image_timing.copy(), kernel_timing.copy(), stride_timing, padding_timing, dilation_timing, groups_timing)
#                     im2col_times.append(time.time() - start_time)

#                     # --- im2patch Conv2d (im2win) - Strided (using strided im2patch as default now) ---
#                     start_time = time.time()
#                     im2patch_conv2d(input_image_timing.copy(), kernel_timing.copy(), stride_timing, padding_timing, dilation_timing, groups_timing, use_strided_im2patch=True) # Strided im2patch - using as default im2patch now
#                     im2patch_times.append(time.time() - start_time) # Changed to im2patch_times


#                 pytorch_time = np.mean(pytorch_times) # Average times
#                 im2col_time = np.mean(im2col_times)
#                 im2patch_time = np.mean(im2patch_times) # Changed to im2patch_time, now using strided im2patch as default

#                 print(f"PyTorch / im2col: {im2col_time / pytorch_time:.2f}x")
#                 print(f"PyTorch / im2patch: {im2patch_time / pytorch_time:.2f}x") # Changed to im2patch_time, now using strided im2patch as default