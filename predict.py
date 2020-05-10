import torch
from torch import nn
import numpy as np
from PIL import Image
from utils import load_checkpoint, process_image
import argparse
import json

def predict(image_path, model, topk, gpu=False):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    if gpu:
        device = 'cuda'
    else:
        device = 'cpu'
    
    img = Image.open(image_path)
    image_tensor = process_image(img).unsqueeze(0)
    
    model.eval()
    model.to(device)
    
    model.idx_to_class = {v:k for k,v in model.class_to_idx.items()}
    
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor).squeeze(0)
        
        outputs = torch.nn.functional.softmax(outputs, dim=0)
        
        topK_indices = (-outputs).argsort()
        topK_probs = outputs[topK_indices][:topk]
        
        topK_classes = [model.idx_to_class[i] for i in topK_indices.cpu().numpy()][:topk]
        
        return list(topK_probs.cpu().numpy()), topK_classes
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Returns the Prediction on a Single Image')
    
    parser.add_argument('path_to_image', type=str, help='Path to image to perform inference')
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint')
    parser.add_argument('--top_k', type=int, default=1, help='Returns top_k predictions')
    parser.add_argument('--category_names', type=str, help='Map categories to real names')
    parser.add_argument('--gpu', action="store_true", default=False, help='Flag to use GPU')
    
    args = parser.parse_args()
    
    # Load the model
    model = load_checkpoint(args.checkpoint)
    
    # Make Predictions
    probs, classes = predict(args.path_to_image, model, args.top_k, gpu=args.gpu)
    print('Predicted Categories:',classes)
    print('Probabilities:',probs)
    
    # Map to real class names if given
    if args.category_names != None:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
            
        names = [cat_to_name[i] for i in classes]
        print('Predicted Class Names:',names)
