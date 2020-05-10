from torchvision import models
import torch
from torch import nn
import numpy as np

def build_model(model_arch, hidden_units):
    
    if model_arch == 'vgg16':
        # Load the pretrained model
        model = models.vgg16(pretrained=True)

        # Freeze the model
        for param in model.parameters():
            param.requires_grad = False

        # Override the classification layer at the end
        model.classifier = nn.Sequential(nn.Linear(25088, hidden_units),
                                         nn.ReLU(),
                                         nn.Linear(hidden_units, 102))
    
        return model
    
    elif model_arch == 'resnet34':
        # Load the pretrained model
        model = models.resnet34(pretrained=True)

        # Freeze the model
        for param in model.parameters():
            param.requires_grad = False

        # Override the classification layer at the end
        model.fc = nn.Sequential(nn.Linear(512, hidden_units),
                                 nn.ReLU(),
                                 nn.Linear(hidden_units, 102))
        
        return model
    
    
def train(model, train_loader, 
          val_loader, criterion,
          optimizer, 
          gpu=False, 
          nb_epochs=2,
          print_every=100):
    
    if gpu:
        device = 'cuda'
    else:
        device = 'cpu'
    
    # push to model to the device (CUDA or CPU)
    model.to(device)
    
    # start epoch
    for current_epoch in range(nb_epochs):
        
        print('Started Epoch {}...\n'.format(current_epoch+1))
        
        # TRAIN
        
        # set the model to train
        model.train()
        
        # loop over the batches of the train loader
        for i, (images, labels) in enumerate(train_loader):
            # move the data to selected device
            images = images.to(device)
            labels = labels.to(device)
            
            # set the optimizer to zero gradients
            optimizer.zero_grad()
            
            # pass the inputs through the model
            outputs = model(images)
            
            # calculate loss
            loss = criterion(outputs, labels)
            # backpropagate
            loss.backward()
            # optimize
            optimizer.step()
            
            # print the losses
            if i % print_every == 0:
                print('Epoch {} Step {} Loss {}'.format(current_epoch+1, i, loss.item()))
        
        # VALIDATION
        
        # set the model to evaluation
        model.eval()
        
        # we don't have to calculate gradients during validation
        with torch.no_grad():
            
            val_losses = []
            num_correct = 0
            
            for i, (images, labels) in enumerate(val_loader):
                # move the data to selected device
                images = images.to(device)
                labels = labels.to(device)

                # pass the inputs through the model
                outputs = model(images)

                # calculate loss
                loss = criterion(outputs, labels)
                val_losses.append(loss.item())
                
                # Get absolute prediction
                pred = outputs.argmax(1)

                # Calculating Accuracy
                correct_tensor = pred.eq(labels.float().view_as(pred))
                correct = np.squeeze(correct_tensor.cpu().numpy())
                num_correct += np.sum(correct)
                
            val_acc = num_correct/len(val_loader.dataset)

            print('\nValidation Loss {} Validation Accuracy {}'.format(sum(val_losses)/len(val_losses), 
                                                                       val_acc))
            
        print('\nEnd of Epoch {}\n\n'.format(current_epoch+1))