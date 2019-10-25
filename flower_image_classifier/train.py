from collections import OrderedDict
import time
import argparse

import torch
from torch import nn
from torch import optim
from torchvision import models

from flower_image_classifier.utils import load_transform_data, save_checkpoint


def build_nn(arch='vgg16', hidden_units=512, gpu=False):
    # Use pre-trained network
    # TODO: Use try catch block, and expand the arch list
    if arch in ['vgg16']:
        model = models.vgg16(pretrained=True)
    else:
        print("This architecture is not supported. Please specify another one.")
        return

    # Freeze the features
    for param in model.parameters():
        param.requires_grad = False

    # Build new classifier
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, 4096)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(0.15)),
        ('fc2', nn.Linear(4096, hidden_units)),
        ('relu2', nn.ReLU()),
        ('dropout2', nn.Dropout(0.15)),
        ('fc3', nn.Linear(hidden_units, 102)),
        ('outpout', nn.LogSoftmax(dim=1))
    ]))
    model.classifier = classifier

    device = torch.device('cuda' if torch.cuda.is_available() and gpu else 'cpu')
    model.to(device)

    return model, device


def train(model, train_loader, val_loader, device, learning_rate=0.01, epochs=10):
    # Define loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    # Start training
    steps = 0
    running_loss = 0
    print_every = 20

    train_losses, val_losses = [], []
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} starts...")

        for inputs, labels in train_loader:
            steps += 1
            print(f"Batch {steps}...")
            inputs, labels = inputs.to(device), labels.to(device)

            start = time.time()

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            print(f"Device = {device}; Time per batch: {(time.time() - start) / 3:.3f} seconds")

            if steps % print_every == 0:
                val_loss = 0
                accuracy = 0
                model.eval()  # disables dropout
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(device), labels.to(device)

                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        val_loss += batch_loss.item()

                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                train_losses.append(running_loss / len(train_loader))
                val_losses.append(val_loss / len(val_loader))

                print(f"Epoch {epoch + 1}/{epochs}.. "
                      f"Train loss: {running_loss / print_every:.3f}.. "
                      f"Validation loss: {val_loss / len(val_loader):.3f}.. "
                      f"Validation accuracy: {accuracy / len(val_loader) * 100:.3f}%"
                      f"")
                running_loss = 0
                model.train()

    return model, val_loss, running_loss, optimizer, print_every


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('data_dir', action='store')
    #     parser.add_argument('checkpoint', action='store')
    parser.add_argument('--save_dir', action='store', dest='save_dir')
    parser.add_argument('--arch', action='store', dest='arch', default='vgg16')
    parser.add_argument('--gpu', action='store_true', dest='gpu')
    parser.add_argument('--hidden_units', action='store', dest='hidden_units', type=int, default=512)
    parser.add_argument('--epochs', action='store', dest='epochs', type=int, default=20)
    parser.add_argument('--learning_rate', action='store', dest='learning_rate', type=int, default=0.001)

    pa = parser.parse_args()

    train_data, train_loader, test_data, test_loader, val_data, val_loader = load_transform_data(pa.data_dir)
    model, device = build_nn(pa.arch, pa.hidden_units, pa.gpu)
    model, val_loss, running_loss, optimizer, print_every = train(model, train_loader, val_loader, device,
                                                                    pa.learning_rate, pa.epochs)

    model.class_to_idx = train_data.class_to_idx
    save_checkpoint(model, pa.epochs, val_loss, optimizer, print_every, val_loader, running_loss, pa.save_dir)

