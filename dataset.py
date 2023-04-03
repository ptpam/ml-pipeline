import torchvision
from torchvision import transforms


def load_data(input_size, data_dir="./data"):
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=data_transforms["train"]
    )

    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=data_transforms["val"]
    )

    return trainset, testset
