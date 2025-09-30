Project & Data

Q: What is the project goal?
A: Classify wildlife species from images.

Q: What’s the key difference between train and test sets in image competitions?
A: Train has images + labels; test has images only.

Q: What is the submission file typically?
A: A CSV with an id column and class probability columns (one per class).

Metrics & Loss

Q: Why use accuracy alongside loss?
A: Loss is hard to interpret; accuracy is intuitive.

Q: What loss is common for multiclass classification?
A: Cross-entropy.

Q: Cross-entropy (log loss) formula?
A: 
LogLoss
=
−
1
𝑁
∑
𝑖
=
1
𝑁
∑
𝑗
=
1
𝑀
𝑦
𝑖
𝑗
 
log
⁡
𝑝
𝑖
𝑗
LogLoss=−
N
1
	​

i=1
∑
N
	​

j=1
∑
M
	​

y
ij
	​

logp
ij
	​


Q: Interpretation of log loss values?
A: Lower is better; penalizes overconfident wrong predictions.

Tensors (Basics)

Q: What is a tensor?
A: An N-dimensional array of values (PyTorch’s core data structure).

Q: What are common tensor attributes to inspect?
A: .shape, .dtype, .device.

Q: What does “color channels” in image tensors mean?
A: Separate planes for colors (e.g., RGB → 3 channels).

Tensors (Creation & Attributes)

Q: How to create a tensor from a Python list of lists?
Code:

import torch
values = [[1, 2, 3], [3, 4, 5], [6, 7, 8]]
t = torch.tensor(values)


Q: How to check dtype and shape of a tensor?
Code:

print(t.dtype)
print(t.shape)


Q: How to check which device a tensor is on?
Code:

print(t.device)

Devices (CPU/GPU)

Q: How to move a tensor to GPU (if available)?
Code:

if torch.cuda.is_available():
    t = t.to("cuda")


Q: What GPU backends are typical on different OSes?
A: CUDA (Windows/Linux), MPS (macOS Apple Silicon).

Q: How to move a model to a device?
Code:

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

Tensor Slicing

Q: Select first two rows of a 2D tensor?
Code:

left = t[:2, :]


Q: Select last two rows?
Code:

right = t[2:, :]


Q: Pick one row / one column / a block?
Code:

row0 = t[0]
col0 = t[:, 0]
block = t[0:2, 1:3]

Tensor Math

Q: Elementwise addition two ways?
Code:

s1 = a + b
s2 = a.add(b)


Q: Elementwise multiplication two ways?
A: Use * or .mul() (note: .mul, not .mull).
Code:

p1 = a * b
p2 = a.mul(b)


Q: Is elementwise multiplication commutative?
A: Yes (ab == ba).

Q: Matrix multiplication two ways?
Code:

c = A @ B
c = torch.matmul(A, B)


Q: When is matrix multiplication valid?
A: Inner dimensions must match; not commutative.

Q: Example result of 
[
2
	
5


7
	
3
]
×
[
8


9
]
[
2
7
	​

5
3
	​

]×[
8
9
	​

]?
A: 
[
61


83
]
[
61
83
	​

].

Aggregations

Q: Mean of all elements in a tensor?
Code:

mean_val = t.mean()


Q: Column-wise vs row-wise mean?
Code:

mean_cols = t.mean(dim=0)  # down columns
mean_rows = t.mean(dim=1)  # across rows

Images & PIL / TorchVision

Q: How to open and view an image with PIL?
Code:

from PIL import Image
img = Image.open(path)
img  # displays in notebooks


Q: Useful PIL image attributes?
A: .size (W,H) and .mode (e.g., "RGB", "L").

Q: Convert PIL image to tensor with TorchVision?
Code:

from torchvision import transforms
tensor_image = transforms.ToTensor()(img)  # CxHxW in [0,1]


Q: What is the image tensor shape convention?
A: [C, H, W].

Q: How to access and plot a single color channel?
Code:

red = img_tensor[0, :, :]


Q: Max/Min over a tensor?
Code:

mx = torch.amax(x)
mn = torch.amin(x)

Preprocessing Pipeline

Q: What does transforms.Compose do?
A: Chains preprocessing steps in order.

Q: Convert to RGB, resize, to tensor pipeline?
Code:

class ConvertToRGB:
    def __call__(self, img):
        return img.convert("RGB") if img.mode != "RGB" else img

transform = transforms.Compose([
    ConvertToRGB(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


Q: Why define __call__ in a class transform?
A: Makes the instance callable like a function in Compose.

Dataset & Class Distribution

Q: Load folder-structured dataset with transform?
Code:

from torchvision import datasets
dataset = datasets.ImageFolder(root=train_dir, transform=transform)

To open the list of folders which function do you use?
datasets.ImageFolder() 
Q: Get distinct class names from ImageFolder?
A: dataset.classes.

Q: Count images per class with Counter + mapping?
Code:

from collections import Counter
import pandas as pd
from tqdm import tqdm

def class_counts(ds):
    counts = Counter(y for _, y in tqdm(ds))
    idx_map = ds.class_to_idx
    return pd.Series({cls: counts[idx] for cls, idx in idx_map.items()})


Q: Bar chart of class distribution (Matplotlib)?
Code:

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(class_counts.index, class_counts.values)
ax.set_xlabel("Class")
ax.set_ylabel("Count")
ax.set_title("Class Distribution")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



Q: Why are GPUs important for building models?

A: GPUs allow models to be built more quickly compared to CPUs, enabling bigger and faster model training.

Q: Do all computers come with GPUs?

A: No, only some computers come with GPUs, which are used for bigger and faster model building.

Q: Which package in PyTorch is used to access GPUs on Linux and Windows machines?

A: The cuda package is used to access GPUs on Linux and Windows.

Q: Which package in PyTorch is used to access GPUs on Macs?

A: The mps package is used to access GPUs on Macs.

Q: How can you check if a GPU (CUDA) is available in PyTorch?

if torch.cude.is_available()


Q: How do you print which device is being used in PyTorch?
Code (Python):

print(f"{device}")

Q: How do you define the path to the main dataset folder called "data_binary" inside "data_p1"?

data_dir = os.path.join("data_p1", "data_binary")


Q: How do you list all labels (folders) inside the training directory in Python?
Code (Python):

labels = os.listdir(train_dir)

Q: In binary classification, how many labels do we expect?
A: Two labels.

Q: How is the training data for each label organized in binary classification?

A: Each label has its own folder containing its data.

Q: What does the folder name represent in the training dataset?

the name of the label 

Q: How do you print the number of "hog" images in the training directory?
Code (Python):

print(len(hog_images))


Q: How do you open an image file in Python using PIL?
Code (Python):

from PIL import Image 
img = Image.open("path")

Q: How do you print the mode (color format) and size of an image in PIL?
Code (Python):

print(img.mode , img.size)

Q: What does .mode represent in a PIL image?
A: The color format of the image (e.g., "RGB", "L").

Q: What does .size represent in a PIL image?
the tupe that showing width and hight of the image 

Q: Why do we need to ensure all images are in the same mode during preprocessing?

maintain consistency 

Q: What mode is commonly required for image datasets in deep learning?
RGB

Q: In PIL, what does the mode "L" represent?
gray scale 

Q: How do you convert an image to RGB mode in PIL?
img = img.convert("RGB")

Q: How can you implement a custom transformer in PyTorch to convert images to RGB if needed?
Code (Python):

class ConvertToRGB:
    def __call__(self, img):
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

Q: What is the purpose of the __call__ method in a Python class?

it makes the class instance callable 

Q: Why do we implement __call__ in custom PyTorch transforms?

A: To allow the transform class to be called directly in preprocessing pipelines (e.g., inside transforms.Compose).

Q: What PyTorch utility is commonly used to build an image preprocessing pipeline?

A: transforms.Compose from torchvision.

A: transforms.Compose from torchvision.
A: Combines multiple preprocessing steps into a single pipeline applied in order.

Q: What are common preprocessing steps in a PyTorch image pipeline?

convert to rgb 
resize the imgages 
convert to tensor 

Q: How do you typically apply a transformation pipeline when loading data in PyTorch?

A: Pass the composed transforms to the dataset so each image is processed before being returned.

Q: How do you define a preprocessing pipeline in PyTorch?
Code (Python):

from torchvision import transforms

transform = transforms.Compose([...])

Q: How do you resize images to 224x224 using torchvision transforms?
Code (Python):

transforms.Resize((224, 224))

Q: How do you convert images to tensors in a PyTorch pipeline?
Code (Python):
transforms.ToTensor()

Q: Which PyTorch utility is commonly used to load image datasets organized in folders?
A: datasets.ImageFolder

Q: How do you load a dataset with ImageFolder in PyTorch?
Code (Python)

from torchvision import datasets

dataset = datasets.ImageFolder(root="path/to/train", transform=transform)

Q: How do you apply preprocessing transforms automatically when loading a dataset with ImageFolder?
A: Pass the transform argument when creating the dataset.

Q: In a PyTorch ImageFolder dataset, what does the attribute .classes return?
a list of distinct classs name 

Q: How are class labels determined in an ImageFolder dataset?
A: From the names of the subfolders inside the root directory.

Q: In a PyTorch ImageFolder dataset, what does the .imgs attribute contain?
A: A list of tuples (image_path, label_number).

Q: What do the numeric label values in .imgs represent?
the classes that we have and assigned by ImageFolder

Q: How can you prove that only two distinct label values exist in your dataset?
Code (Python):
distinct_classes = set(label for _, label in dataset.imgs)
print(distinct_classes)

Q: Why is a set useful for checking distinct label values?
A: A set automatically removes duplicates, leaving only unique labels.


Q: What is a common ratio used to split a dataset into training and validation sets?
A: 80% training, 20% validation.

Q: Which PyTorch function is commonly used to split datasets into subsets?
A: torch.utils.data.random_split


Q: What is the purpose of a DataLoader in PyTorch?
shffling , batching , loading data during trainning 


Q: What is a typical batch size used for training neural networks?
32 

Q: What is the key difference between the DataLoader for training and validation?
trainning need shuffling , validation does not 


Q: Why is shuffling used in training data?
Avoiding too much influecne and impove generalization 

Q: How can you make PyTorch data loading reproducible?
using random generator with fixed seeds 

Q: How do you create a DataLoader for training with shuffling enabled?fixing the random order each run

from torch.utils.data import DataLoader 
import torch 

g = torch.generator().manual_seed(42);

train_loader = DataLoader(
    traine_dataset,
    batch_size = 32,
    shuffle = True , 
    generator = g
)

Q: What is a shallow neural network?
A: A neural network with only a few layers, such as an input layer, two hidden layers, and an output layer.

Q: What is the architecture of the shallow neural network described?
A: Input layer → 2 hidden layers → Output layer.

Q: What is the input shape of each image in the dataset?
3×224×224 (channels, height, width).

Q: Why do we need to flatten images before passing them to a fully connected neural network?

Q: What is lost when we flatten images?

Q: How many features does each flattened image have when resized from 
3
×
224
×
224
3×224×224?

Q: How do you flatten images in PyTorch?

import torch.nn as nn 

flatter = nn.Flatten()
tensor_flatten = flatter(images)

Q: Why is it tedious to manually apply all layers of a neural network in PyTorch?
A: Because you would have to manually pass the data through each layer step by step.

Q: What does nn.Sequential do in PyTorch?
A: It allows you to define a sequence of layers and automatically runs data through them in order.

Q: What are the benefits of using nn.Sequential?
A: Simpler model definition, readability, and automatic layer sequencing.

Q: In the given shallow network, what is the purpose of nn.Flatten() as the first layer?
A: It converts each input image into a 1D vector to be processed by fully connected layers.

create a neurl netowork using nn library 

import torch.nn as nn

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(input_size, 512),
    nn.ReLU(),
    nn.Linear(512, 128),
    nn.ReLU(),
)
Q: What does nn.Linear(in_features, out_features) do in a neural network?
A: Applies a linear transformation 
𝑦
=
𝑊
𝑥
+
𝑏
y=Wx+b to the input.

Q: What is the role of nn.ReLU() in a neural network?
A: Applies the ReLU activation function to introduce non-linearity, keeping positive values and setting negatives to zero.

Q: How can you check the structure of a PyTorch model?
Code:print(type(model))
print(model)

Q: What is the purpose of a loss function in training a neural network?
A: To measure how well the model performs for a given set of parameters.

Q: Which loss function is commonly used for classification tasks in deep learning?
A: Cross-entropy loss (nn.CrossEntropyLoss()).

Q: What is the role of an optimizer in training a neural network?
A: To adjust model parameters to minimize the loss function.

Q: What type of optimizer is Adam?
A: A gradient-based optimizer that improves upon accidental gradient descent.

Q: How do you define the Adam optimizer in PyTorch with a specified learning rate?

Q: How do you define the Adam optimizer in PyTorch with a specified learning rate?

import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=0.001)

Q: What does the lr parameter in an optimizer control?
A: The learning rate, which determines the step size in gradient descent.


Q: What is the purpose of the train_epoch function?
A: To train a model for one epoch over the entire training dataset.

Q: Why do we call model.train() before training?
A: It sets the model to training mode, enabling behaviors like dropout and batch normalization updates.

Q: Why must we call optimizer.zero_grad() before backpropagation?
A: To reset old gradients

Q: Why do we move inputs and targets to the specified device?
A: To ensure computation happens on the same device (CPU or GPU) as the model.

Q: What happens during the forward pass in training?
A: The model processes inputs and outputs

What are the main steps performed inside a train_epoch function in PyTorch?
Teacher studding students lessons with giving them practice 

put everythign in train mode 

loop through all datasets 

reset gradient descent 

moving input / target to device

cal the output 

compare the output and input

imporove the system backward pass

update the weight 

accumulate the batch loss 

return average dataloss 

Q: What are the main steps performed inside a predict function in PyTorch (story form)?
A: Imagine the teacher (the model) is done teaching and now just tests students quietly:

The teacher prepares an empty gradebook (all_probs = []).

The teacher switches to “exam mode” (model.eval()), no more teaching tricks.

The teacher tells themselves not to take notes during grading (torch.no_grad()).

Each group of students (batches) hands in answers (inputs).

The teacher checks answers and gives raw scores (logits).

The teacher then converts scores into clear percentages (softmax → probabilities).

The teacher records all percentages in the gradebook (all_probs).

At the end, the teacher hands back the complete gradebook with all results.

Q: Why do we evaluate models using both loss and accuracy?
A: Loss measures how well predictions fit the data; accuracy measures how many predictions are correct.

Q: What does model.eval() do in PyTorch?
A: Sets the model to evaluation mode (disables dropout, stops batch norm updates).

Q: What does model.train() do in PyTorch?
A: Sets the model to training mode (enables dropout, updates batch norm statistics).

Q: How do you save a model’s parameters in PyTorch?
Code:

torch.save(model.state_dict(), "model.pth")


Q: How do you load saved model parameters in PyTorch?
Code:

model.load_state_dict(torch.load("model.pth"))
model.eval()


Q: Why should you call model.eval() after loading a model?
A: To ensure it runs in evaluation mode for inference (no dropout, stable batch norm).

Q: What is the purpose of a pooling layer in a neural network?
A: To reduce the size of feature maps, lowering computation and dimensions.

Q: How does max pooling work?
A: It takes a 2×2 window, selects the maximum value, and places it in the new feature map.

Q: How does mean pooling work?
A: It takes a 2×2 window and stores the average (mean) value in the new feature map.

Q: What are the two main types of pooling?
A: Max pooling and mean pooling.

Q: What are the benefits of pooling layers?
A:

Reduce computation and dimensions

Reduce overfitting (fewer parameters)

Make the model flexible to variation and distortion

Q: What is the overall process of a neural network for image recognition?
A: Images → Convolution + ReLU → Pooling → Convolution + ReLU → Pooling → Flatten → Fully connected dense layer → Prediction (e.g., "Is this a koala?").

Q: What role does convolution play in a neural network?
A: It extracts features (like shapes) using filters, combining them to form higher-level representations.

Q: What is the purpose of ReLU in CNNs?
A: Introduce non-linearity, speed up training, keep positive values, and set negative values to zero.

Q: What are the three main benefits of convolution operation?
A:

Reduce overfitting

Location-invariant feature detection

ReLU adds non-linearity and speeds up computation

Q: What are the benefits of pooling in CNNs?
A:

Reduce dimensions and computation

Reduce overfitting

Make model tolerant to small distortions and variations

Q: What is a limitation of CNNs?
A: They cannot naturally handle rotation and scale variations.

Q: How can CNNs handle rotation and scale?
A: By using data augmentation (rotating, scaling, and resizing training samples).

Q: What is data augmentation in deep learning?
A: Generating new training samples by rotating, scaling, or modifying original images to improve model robustness.

Q: How are convolutional filters in CNNs learned?
A: The network automatically learns filters during training instead of requiring manual design.

Would you like me to also convert this into a ready-to-import CSV file for Quizlet/Anki so you can upload and start using it immediately?

Q: How do you import core Python utilities for file paths and system info?
Code:

import os
import sys
from collections import Counter

Q: How do you import plotting libraries for visualizing results?
Code:

import matplotlib
import matplotlib.pyplot as plt

Q: How do you import numeric and tabular data tools?
Code:

import numpy as np
import pandas as pd

Q: How do you import PIL for basic image handling?
Code:

import PIL

Q: How do you import PyTorch core modules for models and optimization?
Code:

import torch
import torch.nn as nn
import torch.optim as optim

Q: How do you import torchvision utilities for datasets and transforms?
Code:

import torchvision
from torchvision import datasets, transforms

Q: How do you import data loading helpers for batching and splitting?
Code:

from torch.utils.data import DataLoader, random_split

Q: How do you import a model summary helper for PyTorch?
Code:

from torchinfo import summary

Q: How do you import tools to compute and display a confusion matrix?
Code:

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

Q: How do you add a progress bar for notebook training loops?
Code:

from tqdm.notebook import tqdm

Q: How do you make cuDNN operations deterministic for reproducibility?
Code:

torch.backends.cudnn.deterministic = True