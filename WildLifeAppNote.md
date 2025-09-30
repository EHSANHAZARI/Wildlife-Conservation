📘 PyTorch & Deep Learning Notes
Project & Data

Q: What is the project goal?
A: Classify wildlife species from images.

Q: What’s the key difference between train and test sets in image competitions?
A: Train has images plus labels; test has images only.

Q: What is the submission file typically?
A: A CSV with an id column and class probability columns (one per class).

Metrics & Loss

Q: Why use accuracy alongside loss?
A: Loss is hard to interpret; accuracy is intuitive.

Q: What loss is common for multiclass classification?
A: Cross-entropy.

Q: Cross-entropy (log loss) formula?
A: Log loss equals negative one divided by N times the sum, from i equals one to N, of the sum, from j equals one to M, of y sub i j times the logarithm of p sub i j.

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

import torch
values = [[1, 2, 3], [3, 4, 5], [6, 7, 8]]
t = torch.tensor(values)


Q: How to check dtype and shape of a tensor?

print(t.dtype)
print(t.shape)


Q: How to check which device a tensor is on?

print(t.device)

Devices (CPU/GPU)

Q: How to move a tensor to GPU (if available)?

if torch.cuda.is_available():
    t = t.to("cuda")


Q: What GPU backends are typical on different OSes?
A: CUDA (Windows/Linux), MPS (macOS Apple Silicon).

Q: How to move a model to a device?

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


Q: Why are GPUs important for building models?
A: GPUs allow models to be built more quickly compared to CPUs, enabling bigger and faster model training.

Q: Do all computers come with GPUs?
A: No, only some computers come with GPUs, which are used for bigger and faster model building.

Q: Which package in PyTorch is used to access GPUs on Linux and Windows machines?
A: The cuda package.

Q: Which package in PyTorch is used to access GPUs on Macs?
A: The mps package.

Q: How can you check if a GPU (CUDA) is available in PyTorch?

torch.cuda.is_available()


Q: How do you print which device is being used in PyTorch?

print(f"{device}")

Tensor Slicing

Q: Select first two rows of a 2D tensor?

left = t[:2, :]


Q: Select last two rows?

right = t[2:, :]


Q: Pick one row / one column / a block?

row0 = t[0]
col0 = t[:, 0]
block = t[0:2, 1:3]

Tensor Math

Q: Elementwise addition two ways?

s1 = a + b
s2 = a.add(b)


Q: Elementwise multiplication two ways?
A: Use * or .mul().

p1 = a * b
p2 = a.mul(b)


Q: Is elementwise multiplication commutative?
A: Yes (a multiplied by b equals b multiplied by a).

Q: Matrix multiplication two ways?

c = A @ B
c = torch.matmul(A, B)


Q: When is matrix multiplication valid?
A: Inner dimensions must match; not commutative.

Q: Example result of matrix multiplication

Matrix A equals
two five
seven three

multiplied by vector B equals
eight
nine

The result is a vector:
sixty-one
eighty-three.

Aggregations

Q: Mean of all elements in a tensor?

mean_val = t.mean()


Q: Column-wise vs row-wise mean?

mean_cols = t.mean(dim=0)  # down columns
mean_rows = t.mean(dim=1)  # across rows


Q: Max/Min over a tensor?

mx = torch.amax(x)
mn = torch.amin(x)

Images & PIL / TorchVision

Q: How to open and view an image with PIL?

from PIL import Image
img = Image.open(path)
img  # displays in notebooks


Q: Useful PIL image attributes?
A: .size (width, height) and .mode (e.g., "RGB", "L").

Q: Convert PIL image to tensor with TorchVision?

from torchvision import transforms
tensor_image = transforms.ToTensor()(img)  # CxHxW in [0,1]


Q: What is the image tensor shape convention?
A: [channels, height, width].

Q: How to access and plot a single color channel?

red = img_tensor[0, :, :]