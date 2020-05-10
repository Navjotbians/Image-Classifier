# Imports here
import torch
from torch import nn
from torchvision import models, datasets, transforms
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse

from utils import train_transforms, test_transforms, save_checkpoint
from model import build_model, train


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Trains a Deep Learning Model on Flower Images')
    
    parser.add_argument('data_dir', type=str, help='Path to data directory')
    parser.add_argument('--save_dir', type=str, default='', help='Directory to save the checkpoint to')
    parser.add_argument('--ckpt_name', type=str, default='checkpoint-0.pth', help='The name of the checkpoint')
    parser.add_argument('--arch', type=str, default='vgg16', choices=['vgg16', 'resnet34'], 
                        help='Valid Choices for architecture are vgg16 and resnet34')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units in classifier')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--gpu', action="store_true", default=False, help='Flag to use GPU')
    
    args = parser.parse_args()
    
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'

    # Load the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(valid_dir, transform=test_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16)
    
    # Build the Model
    model = build_model(args.arch, args.hidden_units)
    
    # Define the Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    
    if args.arch == 'vgg16':
        optimizer = torch.optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    elif args.arch == 'resnet34':
        optimizer = torch.optim.Adam(model.fc.parameters(), lr=args.learning_rate)
    
    # Train Model
    train(model, train_dataloader, val_dataloader, 
          criterion, optimizer, gpu=args.gpu, nb_epochs=args.epochs)
    
    # Set the attribute of class_to_idx
    model.class_to_idx = train_dataset.class_to_idx
    
    # Save the checkpoint
    save_checkpoint(model, args.arch, optimizer, args.epochs, args.learning_rate, args.hidden_units, args.save_dir, args.ckpt_name)
    