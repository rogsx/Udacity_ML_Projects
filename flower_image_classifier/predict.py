import torch
import numpy as np
import argparse
from PIL import Image
from flower_image_classifier.utils import process_image, load_checkpoint


def predict(image_dir, category_names, model, topk=5, gpu=False):
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
    parser = argparse.ArgumentParser()

    parser.add_argument('input', action='store')
    parser.add_argument('checkpoint', action='store')
    parser.add_argument('--category_names', action='store', dest='category_names')
    parser.add_argument('--gpu', action='store_true', dest='gpu')
    parser.add_argument('--top_k', action='store', dest='top_k', type=int)

    pa = parser.parse_args()

    checkpoint = load_checkpoint(pa.checkpoint, pa.gpu)
    #     print(checkpoint)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['mapping_to_ind']
    probabilities, top_named_classes = predict(pa.input, pa.category_names, model, pa.top_k, pa.gpu)

    print(probabilities, top_named_classes)

    max_p = max(probabilities)
    max_index = np.argmax(probabilities, axis=0)
    pred_class = top_named_classes[max_index]

    print(f"The predicted class is {pred_class}, and it's probability is {max_p * 100:.3f}%.".format(pred_class, max_p))
