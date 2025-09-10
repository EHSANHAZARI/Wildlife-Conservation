Track animal in wildlife preserve 

The goal is to take picture and know what kind of animal is this 

Crateing more powerful neural network models that take image as input , classify them into multiple category 







How to read image files and preapare for machine learning 

How to use PyTorch to manipulate tensors and build a neural network 

"tensors : what is meaning here "

how to build a convolutional neurla network that works well with images

how to use model to make prediction on new images 

how to run those prediction into submmsiion to competion 





what is the key distinction between train and test dataset in this competition?

&nbsp;

The train data sets contains both **images and labels** , the test data sets contains **only image** 



**what was the goal of the porject ?**	



classify the wildlife species by the picture is our goal

&nbsp;



The test data set contains images and labels of every image



The test is only cotains the images but your task to make prediction on those images 



the images contains the csv file 



id of each image



filepath of image  where it stored



site where the image is taken 



some images are blank and some of them have the picture of the animal 



The file that we are looking to made have 9 columns 

id and type of the animals which we have numbers showing the possibilities of every each animal to be in the picture \\\\



performance metrices 



to measure your model accuracy by looking at prediction error we use metric called error metric so a **lower value is better** oppose to accuracy metric. Log loss can be calculated as follows 



Loss Formula Picture 



N : number of observations 

M : number of classes 

y (ij) The true answer for observation i 

p (ij) model predicted possibility that i belong to class j 



The lower log loss the better probability that the picture is into that category 





Check important attributes of tensors, such as size, data type, and device.

Manipulate tensors through slicing.

Perform mathematical operations with tensors, including matrix multiplication and aggregation calculations.

Download and decompress the dataset for this project.

Load and explore images using PIL.

Demonstrate how visual information is stored in tensors, focusing on color channels.

New Terms:



Attribute

Class

Color channel

Method

Tensor





Def of Tensor : the word tensor comes from mathematic it refest to an array of values that organized into one or more dimension 



for creating and manipulating the tensor we are using the library PyTorch which is built for deep learning 





two things we need 



**what does these library do in python ?**



import os    Lets Python talk to your computer’s files and folders.

import sys   Lets you work with command-line arguments or exit the program.

import matplotlib.pyplot as plt    Makes charts, graphs, and images.

import pandas as pd  Handles tables, CSV/Excel files, databases.

from PIL import Image  Opens, edits, and saves pictures.

import torch  Creates neural networks, tensors (multi-dimensional arrays), and trains AI models.

import torchvision  Includes datasets (like MNIST digits, CIFAR-10 images), image transformations, and prebuilt neural nets.	





**what does it mean that torch creates neural network ?**



it means that it creates a simple layers to make decision 





Pytroch tensors 





**how to transform a multidimension array to tensor give an example :**



how to transform the array of values into tensors 

&nbsp; 

my\_values = \[\[1,2,3] , \[3,4,5] , \[6,7,8]]

my\_tensor = torch.Tensor(my\_values)

print(my\_tensor)



"shows the output please"



**what is the data type of tensor and give me some functionality that it can use of its benefit :**



Note that the python tensor is a class , like any class it has some attributes that can help us , like shape can give us the dimension of the tensor 

.dtype gives us the data type of the values stores in tensor 





**how can we find out the type of the attributes in the tensor and how can we find out the dimension of the tensor ?**



tensor have a class Tensor() 



print(my\_tensor.dtype)

print(my\_tensor.shape)







**how can we know on which hardware the tensor is stored ?**



tensor also have .device attributes which can tell us that on which hardware the tensor is stored 







**how in pytorch we can get access to gpu ?**



In PyTorch, CUDA is used to access the GPU on Linux and Windows machines, while MPS is used for GPU acceleration on Macs.







**how to move our tensor from cpu to gpu give a code  ?** 



we have to move our tensor from cpu to gpu 



my\_tensor = my\_tensor.to("cuda")

my\_tensor.device is now showing cuda 









/////// 



There are several ways to manipulate tensors



**slicing :**we use square brackets and index to select a subset of the value in tensor 



**# to get the first two rows** 

left\_tensor = my\_tensor\[:2 , :] first element is rows and second element is column 



\#to select the last two rows

right\_tensor = my\_tensor\[2: , :]



tensor = torch.tensor(\[\[1, 2, 3, 4],

&nbsp;                      \[5, 6, 7, 8],

&nbsp;                      \[9,10,11,12]])



**question # 1. Pick one row**

print(tensor\[0])         # → \[1, 2, 3, 4]



**question # 2. Pick one column**

print(tensor\[:, 0])      # → \[1, 5, 9]



**question # 3. Pick a first 2 rows**

print(tensor\[0:2])       # → first 2 rows



**question # 4. Pick a block (submatrix)**# → rows 0–1, cols 1–2

print(tensor\[0:2, 1:3])  # → rows 0–1, cols 1–2





**how can you perform adding of tensor ?**



**mathematical operations :** we can peform adding using the + operator or the add() method 



summed\_tensor\_operator = left\_tensor + right\_tensor

summend\_tensor\_method = left\_tensor.add(right\_tensor)



Left tensor:

&nbsp;tensor(\[\[1, 2],

&nbsp;        \[3, 4]])



Right tensor:

&nbsp;tensor(\[\[10, 20],

&nbsp;        \[30, 40]])



Sum with + :

&nbsp;tensor(\[\[11, 22],

&nbsp;        \[33, 44]])



Sum with .add():

&nbsp;tensor(\[\[11, 22],

&nbsp;        \[33, 44]])



**elementwise multiplication**  

use \* or .mull() function to do that 



ew\_operator = left\_tensor \* right\_tensor 

ew\_method = left\_Tensor.mull(right\_tensor)



Point : its commutative it means either right \* left == left \* right 



Left tensor:

&nbsp;tensor(\[\[1, 2],

&nbsp;        \[3, 4]])



Right tensor:

&nbsp;tensor(\[\[10, 20],

&nbsp;        \[30, 40]])



Elementwise \* :

&nbsp;tensor(\[\[10, 40],

&nbsp;        \[90, 160]])



Elementwise .mul():

&nbsp;tensor(\[\[10, 40],

&nbsp;        \[90, 160]])



Right \* Left:

&nbsp;tensor(\[\[10, 40],

&nbsp;        \[90, 160]])





**Matrix Multiplication** 



you can use @ or matmul method , if the number of rows in one of them is equal to number of column in the other they can be multiplied 





tensor(\[2., 5.] , \[7. , 3.]) @ tensor(\[8.],\[9.]])



get the first row of A  get the column of B 



each \[] : 

each number is element of rows 



**find the answer of the below multiplication :**



A = torch.tensor(\[\[2, 5],

&nbsp;                 \[7, 3]])



B = torch.tensor(\[\[8],

&nbsp;                 \[9]])



result = torch.matmul(A, B)  # or A @ B

print(result) 

tensor(\[\[61],

&nbsp;       \[83]])



**under which circumstance the multiplication of the two matrix is possible ?**



matrix multiplication is not commutative and if the rows of one matrix is not equal to column of the other one its not possible 





what is the meaning of commutative ? the left to right is equal right to left





**aggregation caclculation** 



my\_tenso.mean() create a tensor with the mean value of all the tensor elements 



import torch



**calculate the mean of the below tensor and which function does that ?**





\# Create a tensor

my\_tensor = torch.tensor(\[\[2., 4.], 

&nbsp;                         \[6., 8.]])



\# Take the mean

mean\_value = my\_tensor.mean()

print(mean\_value)



this prints 5. as the result of the mean 





we can cacluate the mean of the rows or columns in tensor , with dim =  argument it shows in which direction we want to calculate the mean of 



the first value is representing the column 0  and the second value representing the rows 1 



import torch



tensor = torch.tensor(\[\[2., 4.],

&nbsp;                      \[6., 8.]])



**# Mean of each column** (dim=0 → downwards)

mean\_col = tensor.mean(dim=0)



**# Mean of each row** (dim=1 → across)

mean\_row = tensor.mean(dim=1)



print("Original tensor:\\n", tensor)

print("Mean by columns (dim=0):", mean\_col)

print("Mean by rows (dim=1):", mean\_row)







Lets using our dataset 



\# -------------------------------

\# Google Cloud Platform (GCP) CLI Commands

\# -------------------------------



\# 1. List files in Google Cloud Storage

gcloud storage ls



\# (In Jupyter notebooks, prefix with "!" to run shell commands)

\# Example: !gcloud storage ls



\# 2. Download a file from GCP bucket into your VM

gcloud storage cp "gs://bucket-name/project1.tar.gz" . --no-clobber



\# --no-clobber : prevents downloading again if the file already exists



\# 3. Decompress (extract) the downloaded file

tar --skip-old-files -xzf project1.tar.gz



\# --skip-old-files : avoids overwriting files that are already extracted

\# -x : extract

\# -z : handle gzip format

\# -f : specify file name



\# 4. For more options and help with tar

tar --help







in our code we have two files train and test which in train file we have the picture and title of the animal where there are files with the name of the animals and the picture contains the animal 



we can use os.path.join to make our data)Dir 



data\_dir = os.path.join("data\_p1" , "data\_multiclass")

train\_dir = os.path.join(data\_dir , "train")





now we have path to train\_dir 



investigation the content 



create a list of the contest of train\_Dir and assign the result to class\_directories



class\_directory = os.listdir(train\_dir)

print(class\_directory)





**class distribution mapping :**

we have to create path for every each subdirectory that is in that class directory and then pass it to a dictionary 





class\_distribution\_dict = {}



for subdirectory in class\_directory :

&nbsp;	dir = os.path.join(train\_dir , sub\_directory)

&nbsp;	files = os.listdir(dir)

&nbsp;	num\_files = len(files)

&nbsp;	class\_distribution\_dict\[subdirectory] = num\_files;



how to create a bar chart 



x\_names = \[]

y\_names = \[]



define the fig and ax 



fig , ax = plt.subplots()

bar\_containers = ax.bar(x\_names , y\_names)



setting the y labeles title of the chart and min and max

ax.set (y\_labels = "y lables" , title = "title of the chart" , ylim = (0 ,5000)

defining the lable of the bar 

ax.bar\_label (bar\_Containers , fmt ='{: , .0f}'





This is example 



\# Create a bar plot of class distributions

fig, ax = plt.subplots(figsize=(10, 5))



\# Plot the data

ax.bar(class\_distributions.keys() , class\_distributions.values()) # Write your code here

ax.set\_xlabel("Class Label")

ax.set\_ylabel("Frequency \[count]")

ax.set\_title("Class Distribution, Multiclass Training Set")

plt.xticks(rotation=45)

plt.tight\_layout()

plt.show()





**PILLOW library**



allows up to open and see the images 



hot\_image\_pil = Image.open(hot\_image\_path) 



print("the data type of the image : " , type(hot\_image\_pil))

hot\_image\_pil   #it displays the image



**Image Attribution** 



.size give us the size of height and width of the image 



.mode if it is colord image or black and white RGB or L



we have loaded two images files using pillow library  we were using JpegImageFiles, we need to represent them as tensors



The pytorch community created the torchvision library which comes with a lot of helpful tools to get information form image 



red\_channel = antelope\_tensor\[0, :, :]

ax0.imshow(red\_channel, cmap="Reds")

ax0.set\_title("Antelope, Red Channel")

ax0.axis("off")



\# Plot green channel

green\_channel = antelope\_tensor\[1, :, :]

ax1.imshow(green\_channel, cmap="Greens")

ax1.set\_title("Antelope, Green Channel")

ax1.axis("off")



\# Plot blue channel

blue\_channel = antelope\_tensor\[2, :, :]

ax2.imshow(blue\_channel, cmap="Blues")

ax2.set\_title("Antelope, Blue Channel")

ax2.axis("off")





Load Tensors 



the pythorch community have library torchivision and we can use ToTensor() to conver image to tensor 



tensor\_image = tranforms.ToTensor()(image\_name) it means that we call this 



when tensor have 3 attributes its 3 dimensional \[a , b ,c ] the first number is color channel second number is pxel and third is the width 



\[C H W]



for example if a picture is gray and balck and white its 1 

when its 3 it has 3 channel rgb 



this is how to access red channel 

red\_channel = antelope\_tensor\[0, :, :]

ax0.imshow(red\_channel, cmap="Reds")

ax0.set\_title("Antelope, Red Channel")

ax0.axis("off")





To find the max and min number in tensor we are using two functions 



max\_channel\_value = tensor.amax(antelope\_tensor)

min\_channel\_value = tensor.amin(antelope\_tensor)



////////



what python traceback 



traceback is report genearted when error occurs in your code



offer valuable information about what and where eror happened 





what is neural network



imagine you are teaching computer how to recognize hand written number like distinguishing 3 from 8 neural network is a simple way of how our brain works 



**Neurons** a small box that holds number between 0 and 1 



the input is bunch of these neurons one for each pixed in the image show it so its totally of 784 pixels



the network is built from layers the input layers followed by 1 or more hidden layers and ending with output layers 



each input layer transform and and refine the information further 



neurons are connected like a web weight or w shows how strong they are connected 



biases is a number added to find the best result 



neurons convers inuts into outputs by calculating all the wighted sum of input adding bias and then passing through activation function like squashing function to keep the result betwween 0 and 1 



how it learns a simple example 



you begin with random weight and biases the network can't recognize anythnig yet 



trains the examples 

show the network many images of handwritten digits each labeled with correct number 



if its wrong the network adjusts weight and biases the output neuron that thinks it see the corret number gets evaluated 



this is done by gradient descent which reduce the mistake over time 



eventually the network is able to identify new unseen digits 





**Gradient Descent in neural network**



gradient descent helps us by calcutating the cost (error) function to each step getting close to what actually the w and b should be to get to the best funciton 







