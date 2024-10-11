from typing import Union

from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100

def get_color_distortion(distortion_strength: float = 1.0):
    """
    Functions to apply color distortion to the images.
    """

    color_jitter = transforms.ColorJitter(
        0.8*distortion_strength, 
        0.8*distortion_strength, 
        0.8*distortion_strength, 
        0.2*distortion_strength
        )
    
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    
    return [rnd_color_jitter, rnd_gray]

class Augment:
    """
    Takes in an image and creates two views of it.

    Parameters
    ----------
    transform: transforms.Compose
        The transformaton pipeline for the images.

    Returns
    -------
    x1: torch.Tensor
        The first view of the image.

    x2: torch.Tensor
        The second view of the image.
    """

    def __init__(self, transform: transforms.Compose):
        self.transform = transform

    def __call__(self, img: Image):
        x1, x2 = self.transform(img), self.transform(img)
        return x1, x2


def get_pretraining_dataset(
    data_dir: str, 
    dataset: str, 
    distortion_strength: float = 0.5
    ) -> Union[CIFAR10, CIFAR100]:

    """
    Constructs the pre-training dataset for contrastive learning.

    Parameters
    ----------
    data_dir: str
        The directory to save the images.

    dataset: str
        Must be one of [`cifar10`, `cifar100`].

    distortion_strength: float
        The strength of the clor distortion.

    Returns
    -------
    pretrain_dataset: Union[CIFAR10, CIFAR100]
        The pre-training dataset for contrastive learning.
    """

    def get_transforms(mean, std):    
        transform = transforms.Compose([
            transforms.RandomResizedCrop(size=(32, 32)),
            transforms.RandomHorizontalFlip(p=0.5),
            *get_color_distortion(distortion_strength),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        return transform

    if dataset == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        transform = get_transforms(mean, std)

        pretrain_dataset = CIFAR10(data_dir, train=True, transform=Augment(transform), download=True)
    
    if dataset == "cifar100":
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        transform = get_transforms(mean, std)

        pretrain_dataset = CIFAR100(data_dir, train=True, transform=Augment(transform), download=True)

    return pretrain_dataset


def get_finetune_dataset(
    data_dir: str, 
    dataset: str
    ) -> Union[CIFAR10, CIFAR100]:

    """
    Constructs the finetuning dataset.
    
    Parameters
    ----------
    data_dir: str
        The directory to save the images.

    dataset: str
        Must be one of [`cifar10`, `cifar100`].

    Returns
    -------
    train_dataset: Union[CIFAR10, CIFAR100]
        The train set for finetuning.

    val_dataset: Union[CIFAR10, CIFAR100]
        The validation set for finetuning.
    """
    
    valid_datasets = ["cifar10", "cifar100"]
    assert dataset in valid_datasets, f"Dataset must be one of {valid_datasets}"

    def get_transforms(mean, std):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        return transform
    
    if dataset == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        transform = get_transforms(mean, std)

        train_dataset = CIFAR10(data_dir, train=True, transform=transform, download=True)
        val_dataset = CIFAR10(data_dir, train=False, transform=transform, download=True)
    
    if dataset == "cifar100":
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        transform = get_transforms(mean, std)

        train_dataset = CIFAR100(data_dir, train=True, transform=transform, download=True)
        val_dataset = CIFAR100(data_dir, train=False, transform=transform, download=True)

    return train_dataset, val_dataset
