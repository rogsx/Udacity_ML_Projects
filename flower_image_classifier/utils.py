from torchvision import datasets, transforms
import torch
import numpy as np


def load_transform_data(data_dir='flowers'):

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

    # Using the image datasets and the transforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=64)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64)

    return train_data, train_loader, test_data, test_loader, val_data, val_loader


def save_checkpoint(model, epoch, val_loss, optimizer, print_every, val_loader, running_loss, checkpoint_path):
    checkpoint = {'model': model,
                  'mapping_to_ind': model.class_to_idx,
                  'state_dict': model.state_dict(),
                  'epochs': epoch,
                  'val_loss': val_loss / len(val_loader),
                  'train_loss': running_loss / print_every,
                  'optimizer_state': optimizer.state_dict()}

    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(checkpoint, gpu=False):
    checkpoint = torch.load(checkpoint, map_location=('cuda' if (gpu and torch.cuda.is_available()) else 'cpu'))
    return checkpoint


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # Resize image while keeping aspect ratio

    width, height = image.size
    aspect_ratio = float(width / height)
    if aspect_ratio >= 1:
        scaled_height = 256
        scaled_width = int(scaled_height * aspect_ratio)
    else:
        scaled_width = 256
        scaled_height = int(scaled_width / aspect_ratio)

    image = image.resize((scaled_width, scaled_height))

    # Crop the image for the center 224 X 224 portion
    left = scaled_width / 2 - 112
    upper = scaled_height / 2 - 112
    right = scaled_width / 2 + 112
    lower = scaled_height / 2 + 112

    image_crop = image.crop((left, upper, right, lower))

    # Color channel scaling
    img_array = np.array(image_crop)
    np_image = img_array / 255.

    # Normalization
    mean = np.array([0.485, 0.456, 0.406])
    std_dev = np.array([0.229, 0.224, 0.225])
    norm_img = (np_image - mean) / std_dev

    # make color channel first dimension
    processed_img = norm_img.transpose((2, 0, 1))

    return processed_img


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax

