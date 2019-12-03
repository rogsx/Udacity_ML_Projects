
# Deep Learning
## Project: Classifying Flower Images

### Install

This project requires **Python 3.x** and the following Python libraries installed:

- [NumPy](https://www.numpy.org/)
- [PyTorch](https://pytorch.org/)
- [TorchVision](https://pytorch.org/docs/stable/torchvision/index.html)

You will also need to have software installed to run and execute an [iPython Notebook](http://ipython.org/notebook.html)

We recommend you install [Anaconda](https://www.continuum.io/downloads), a pre-packaged Python distribution that contains all of the necessary libraries and software for this project. 

### Run

In a terminal or command window, navigate to the top-level project directory `Udacity_ML_Projects/` (that is the parent directory of this README) and run following commands in sequence to train and then predict.

This will train the model in cpu or gpu (if gpu option is specified), save a checkpoint file, and then predict the flower for specified image. 

```bash
python flower_image_classifier/train.py 'flower_image_classifier/flowers/' --save_dir 'chk_pt_temp.pth' --epochs 1 --gpu
```  
Mandatory arguments are:
* `data_dir`: directory containing training, validation and test sets

Available options for the training command line are:
* `save_dir`: checkpoint location for the model, weights and biases
* `arch`: architecture of pre-trained neural net, available architectures include VGG16, #TODO add more archs
* `hidden_units`: the number of input size for the last hidden layer 
* `epochs`: number of training rounds
* `learning_rate`: learning_rate of the optimizer
* `gpu`: whether to use GPU for training, if available 

and

```bash
python flower_image_classifier/predict.py 'flower_image_classifier/flowers/test/1/image_06743.jpg' 'chk_pt_temp.pth' --category_names 'flower_image_classifier/cat_to_name.json' --top_k 5
```
Mandatory arguments are:
* `input`: file path for the image that you want to predict
* `checkpoint`: location of the checkpoint file saved by the training process

Available options for the prediction command line are:
* `category_names`: file path for the cat_to_name mapping file
* `top_k`: the highest k output classes with highest probabilities
* `gpu`: whether to use GPU for training, if available  

### Data

The data contains colored flower images saved in jpg format. The images comes in different sizes
with varying background. They are stored in different classes folders, each containing multiple samples
of the same class. 

The entire dataset was split into training, testing, and validation sets. 