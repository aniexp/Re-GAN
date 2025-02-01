import torch
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from math import log2

def get_loader(image_size, workers, bs, dataroot, data_ratio):
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5 for _ in range(3)],
                [0.5 for _ in range(3)],
            ),
        ]
    )
    batch_size = bs[int(log2(image_size / 4))]  # Calculate the batch size based on image size
    # CIFAR10 dataset loading, including the download option
    dataset = datasets.CIFAR10(root=dataroot, transform=transform, download=True)
    
    # Ensure the subset size is valid (between 0 and 1)
    data_len = len(dataset)
    subset_len = int(data_len * data_ratio)
    subset_len = max(1, subset_len)  # Ensure at least 1 sample is included in the subset

    # Create the subset of the dataset
    subset = torch.utils.data.Subset(dataset, np.arange(subset_len))
    
    # DataLoader for batching the data
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
    )
    return loader, dataset

def gradient_penalty(critic, real, fake, alpha, train_step, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    beta = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * beta + fake.detach() * (1 - beta)
    interpolated_images.requires_grad_(True)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images, alpha, train_step)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty
