import torch
import numpy as np
import math
from torch import nn
from torch.nn import functional as F

class CascadeGaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing separately for each channel (depthwise convolution).

    Arguments:
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.

    """
    def __init__(self, kernel_size, sigma):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size]

        cascade_coefficients = [0.5, 1.0, 2.0]  # std multipliers, hardcoded to use 3 different Gaussian kernels
        sigmas = [[coeff * sigma, coeff * sigma] for coeff in cascade_coefficients]  # isotropic Gaussian

        self.pad = int(kernel_size[0] / 2)  # assure we have the same spatial resolution

        # The gaussian kernel is the product of the gaussian function of each dimension.
        kernels = []
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
        for sigma in sigmas:
            kernel = torch.ones_like(meshgrids[0])
            for size_1d, std_1d, grid in zip(kernel_size, sigma, meshgrids):
                mean = (size_1d - 1) / 2
                kernel *= 1 / (std_1d * math.sqrt(2 * math.pi)) * torch.exp(-((grid - mean) / std_1d) ** 2 / 2)
            kernels.append(kernel)

        gaussian_kernels = []
        for kernel in kernels:
            # Normalize - make sure sum of values in gaussian kernel equals 1.
            kernel = kernel / torch.sum(kernel)
            # Reshape to depthwise convolutional weight
            kernel = kernel.view(1, 1, *kernel.shape)
            kernel = kernel.repeat(3, 1, 1, 1)
            kernel = kernel.to(device)

            gaussian_kernels.append(kernel)

        self.weight1 = gaussian_kernels[0]
        self.weight2 = gaussian_kernels[1]
        self.weight3 = gaussian_kernels[2]
        self.conv = F.conv2d

    def forward(self, input):
        input = F.pad(input, [self.pad, self.pad, self.pad, self.pad], mode='reflect')

        # Apply Gaussian kernels depthwise over the input (hence groups equals the number of input channels)
        # shape = (1, 3, H, W) -> (1, 3, H, W)
        num_in_channels = input.shape[1]
        grad1 = self.conv(input, weight=self.weight1, groups=num_in_channels)
        grad2 = self.conv(input, weight=self.weight2, groups=num_in_channels)
        grad3 = self.conv(input, weight=self.weight3, groups=num_in_channels)

        return (grad1 + grad2 + grad3) / 3

IMAGENET_MEAN_1 = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
IMAGENET_STD_1 = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
LOWER_IMAGE_BOUND = torch.tensor((-IMAGENET_MEAN_1 / IMAGENET_STD_1).reshape(1, -1, 1, 1))
UPPER_IMAGE_BOUND = torch.tensor(((1 - IMAGENET_MEAN_1) / IMAGENET_STD_1).reshape(1, -1, 1, 1))


def generate_novel_domain_perturbation(model, image_tensor, class_text_features, domain_text_features, class_label, domain_label, lr, total_iteration, smoothing_coefficient, smooth, normalize):
    for curr_iteration in range(total_iteration):
        # Step 0: Feed forward pass
        image_features = model.encode_image(image_tensor)
        
        loss = -torch.log((image_features @ class_text_features.T)[torch.arange(image_features.size(0)), class_label]) - torch.log((image_features @ domain_text_features.T)[torch.arange(image_features.size(0)), domain_label])

        loss.backward()

        # Step 3: Process image gradients (smoothing + normalization, more an art then a science)
        grad = image_tensor.grad.data

        # Applies 3 Gaussian kernels and thus "blurs" or smoothens the gradients and gives visually more pleasing results
        # We'll see the details of this one in the next cell and that's all, you now understand DeepDream!
        if smooth:
            sigma = ((curr_iteration + 1) / total_iteration) * 2.0 + smoothing_coefficient
            grad = CascadeGaussianSmoothing(kernel_size=9, sigma=sigma)(grad)  # "magic number" 9 just works well

        # Normalize the gradients (make them have mean = 0 and std = 1)
        # I didn't notice any big difference normalizing the mean as well - feel free to experiment
        if normalize:
            g_std = torch.std(grad)
            g_mean = torch.mean(grad)
            grad = grad - g_mean
            grad = grad / g_std

        # Step 4: Update image using the calculated gradients (gradient ascent step)
        image_tensor.data -= lr * grad

        # Step 5: Clear gradients and clamp the data (otherwise values would explode to +- "infinity")
        image_tensor.grad.data.zero_()
        image_tensor.data = torch.max(torch.min(image_tensor, UPPER_IMAGE_BOUND), LOWER_IMAGE_BOUND)
    return image_tensor
