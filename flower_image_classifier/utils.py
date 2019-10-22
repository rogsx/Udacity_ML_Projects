from torchvision import datasets, transforms, models
import torch


def load_transform_data(train_dir, valid_dir, test_dir):
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                          transforms.RandomRotation(40),
                                          transforms.RandomVerticalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    val_data = datasets.ImageFolder(valid_dir, transform=test_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle= True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=64)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64)

    return train_data, train_loader, test_data, test_loader, val_data, val_loader

