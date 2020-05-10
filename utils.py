import torch
from torch import nn
from torchvision import datasets, transforms
import os
from model import build_model

# Creating the transformations
train_transforms = transforms.Compose([transforms.RandomResizedCrop(256),
                                      transforms.RandomRotation(30),
                                      transforms.RandomHorizontalFlip(), 
                                      transforms.CenterCrop((224,224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

test_transforms = transforms.Compose([transforms.Resize(256), 
                                      transforms.CenterCrop((224,224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

def save_checkpoint(model, arch, optimizer, epochs, learning_rate, hidden_units, save_dir, ckpt_name):
    checkpoint = {'model_state': model.state_dict(),
                  'arch': arch,
                  'optimizer_state': optimizer.state_dict(),
                  'epochs':epochs,
                  'learning_rate': learning_rate,
                  'hidden_units': hidden_units,
                  'class_to_idx': model.class_to_idx
                  }
    
    save_path = os.path.join(save_dir, ckpt_name)
    torch.save(checkpoint, save_path)
    print(f'Checkpoint Saved at {save_path}')
    
def load_checkpoint(ckpt_path):
    # Load the Checkpoint Dict
    checkpoint = torch.load(ckpt_path)
    # Rebuild the Model
    model = build_model(checkpoint['arch'], checkpoint['hidden_units'])
    # Load the weights
    model.load_state_dict(checkpoint['model_state'])
    # Load the class_to_idx dictionary
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model
    
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns a PyTorch Tensor
    '''
    image_tensor = test_transforms(image)
    
    return image_tensor


    