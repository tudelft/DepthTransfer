import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

def add_gaussian_noise(depth_image, mean=0, std=1):
    noise = np.random.normal(mean, std, depth_image.shape)
    noisy_depth_image = depth_image + noise
    return noisy_depth_image

def add_quantization_noise(depth_image, levels=256):
    quantized_depth_image = np.round(depth_image / (255 / (levels - 1))) * (255 / (levels - 1))
    return quantized_depth_image

def apply_bilateral_filter(depth_image, d=5, sigma_color=75, sigma_space=75):
    # Ensure the depth image is a single-channel 32-bit float image
    if len(depth_image.shape) == 3 and depth_image.shape[2] == 3:
        # Convert to grayscale if it's a 3-channel image
        depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY)
    depth_image_float = depth_image.astype(np.float32)
    # Apply the bilateral filter
    filtered_depth_image = cv2.bilateralFilter(depth_image_float, d, sigma_color, sigma_space)
    return filtered_depth_image

def add_occlusion_artifacts(depth_image, occlusion_percentage=0.1):
    occlusion_mask = np.random.rand(*depth_image.shape) < occlusion_percentage
    occluded_depth_image = np.copy(depth_image)
    occluded_depth_image[occlusion_mask] = 0  # Assume occlusion areas have zero depth
    return occluded_depth_image

def introduce_disparity_errors(depth_image, max_error=1.0):
    disparity_error = np.random.uniform(-max_error, max_error, depth_image.shape)
    depth_image_with_errors = depth_image + disparity_error
    return depth_image_with_errors

def simulate_depth_consistency_issues(depth_image, consistency_factor=0.9):
    inconsistent_depth_image = depth_image * consistency_factor + (1 - consistency_factor) * np.random.random(depth_image.shape)
    return inconsistent_depth_image

def add_edge_artifacts(depth_image, edge_width=3, edge_noise_std=2.0, set_edge_to_zero=True):
    # Detect edges using Canny edge detector
    edges = cv2.Canny(depth_image.astype(np.uint8), 100, 200)
    # Dilate edges to create thicker edge regions
    kernel = np.ones((edge_width, edge_width), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel)
    
    # Optionally set edge regions to zero
    if set_edge_to_zero:
        depth_image[dilated_edges > 0] = 0
    else:
        # Add significant noise to edge regions
        noisy_edges = np.random.normal(0, edge_noise_std, depth_image.shape)
        edge_artifacts = np.where(dilated_edges > 0, noisy_edges, 0)
        depth_image += edge_artifacts
    
    return depth_image

def add_motion_blur(depth_image, kernel_size=3, angle=0):
    # Create a motion blur kernel
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1)/2), :] = np.ones(kernel_size)
    kernel = cv2.getRotationMatrix2D((kernel_size / 2 - 0.5, kernel_size / 2 - 0.5), angle, 1)
    kernel = cv2.warpAffine(np.ones((kernel_size, kernel_size)), kernel, (kernel_size, kernel_size))
    kernel = kernel / kernel_size

    # Apply the kernel to the depth image
    blurred_depth_image = cv2.filter2D(depth_image.astype(np.float32), -1, kernel)
    return blurred_depth_image

# load png image
img = Image.open('gray_image_0_1716552479.5455325.png').convert('L')
plt.imshow(img)
plt.show()
# convert image to numpy array
depth_image = np.array(img)
# depth_image = np.load('gray_image_0_1716552479.5455325.png')  # Load your depth image

noisy_depth_image = add_gaussian_noise(depth_image)
quantized_depth_image = add_quantization_noise(noisy_depth_image)
filtered_depth_image = apply_bilateral_filter(quantized_depth_image)
depth_image_with_disparity_errors = introduce_disparity_errors(filtered_depth_image)
depth_image_with_edge_artifacts = add_edge_artifacts(depth_image_with_disparity_errors)
motion_blurred_depth_image = add_motion_blur(depth_image_with_edge_artifacts)
occluded_depth_image = add_occlusion_artifacts(motion_blurred_depth_image)

final_depth_image = simulate_depth_consistency_issues(occluded_depth_image)

# Save or display the final simulated depth image
final_depth_image = final_depth_image.astype(np.uint8)
final_depth_image = Image.fromarray(final_depth_image)
plt.imshow(final_depth_image)
plt.show()

