import torch
import numpy as np
from PIL import Image
from flower_image_classifier.utils import process_image, load_checkpoint


def predict(image_dir, category_names, model, topk=5, gpu='gpu'):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    import json

    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() and gpu else 'cpu')
    model.to(device)

    img = Image.open(image_dir)
    image = process_image(img)
    image_tensor = torch.tensor(image, dtype=torch.float)
    image_tensor = image_tensor.unsqueeze_(0)
    with torch.no_grad():
        logps = model(image_tensor)
    ps = torch.exp(logps)
    probabilities, indices = ps.topk(topk)

    index_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_classes = [index_to_class[i.item()] for i in indices.squeeze()]
    probabilities = np.array(probabilities[0])
    top_named_classes = [cat_to_name[c] for c in top_classes]

    return probabilities, top_named_classes


if __name__ == '__main__':

    checkpoint = load_checkpoint(checkpoint, gpu)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    probabilities, top_named_classes = predict(input, category_names, model, topk, gpu)

    print(probabilities, top_named_classes)