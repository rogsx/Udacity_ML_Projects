import matplotlib.pyplot as plt
from collections import OrderedDict
import time
import argparse

import torch
from torch import nn
from torch import optim
import torch.nn.functional as f
from torchvision import datasets, transforms, models


def build_nn(arch='vgg16', hidden_units=512):
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
        ('fc2', nn.Linear(4096, 1000)),
        ('relu2', nn.ReLU()),
        ('dropout2', nn.Dropout(0.15)),
        ('fc3', nn.Linear(1000, 102)),
        ('outpout', nn.LogSoftmax(dim=1))
    ]))
    model.classifier = classifier

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    return model


def train(model, train_loader, val_loader, device, learning_rate=0.01, epochs=20):
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
                      f"Validation accuracy: {accuracy / len(val_loader):.3f}"
                      f"")
                running_loss = 0
                model.train()
    return model

