import torch
import torchvision


def get_dataloaders(batch_size):

    # Get data
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    train_dataset = torchvision.datasets.MNIST(root="~/wrkshop-demo/datasets", train=True, transform=transforms, download=True)
    val_dataset = torchvision.datasets.MNIST("~/wrkshop-demo/datasets", train=False, transform=transforms, download=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader