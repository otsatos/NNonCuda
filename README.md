# Deep Neural Network with CUDA
Creating DNN building blocks with **C++** code and **Cuda** libraries

## Project Description
Libraries and frameworks like Tensorflow, Pytorch, Deeplearning4j, Keras etc, have revolutionized the data science field with their ready-to-use machine learning tools.
The majority of developers use these libraries and frameworks mainly in Python to solve practical real world problems.
Inspired by the NumPy numerical library in Python, I wanted to create a similar way of performing matrix manipulations and operations when writing code in C++ 
and solving machine learning classification problems. There's nothing new here but a personal interest and intention to explore the data science techniques with Cuda libraries that give access to high-performance computing with little complexity and workarounds. 

The purpose of this project is to provide a **C++** implementation of fundamental deep learning building blocks, 
such as matrix data structure, hidden layers, activation and loss functions, 
that can be used with the same convenience as Python libraries to perform feedforward and backpropagation techniques.
In this project we are dealing with a classification problem using the Iris Data set. 
The **Iris Dataset** is a well-known dataset in machine learning that consists of 150 samples of iris flowers, each of which is described by four features: sepal length, sepal width, petal length, and petal width. 
The goal of the problem is to classify each sample into one of three possible species: Iris-setosa, Iris-versicolor, or Iris-virginica.

## The solution
We have developed a program in C++ that leverages the power of cuBLAS and Cuda runtime libraries to accelerate 
the training of a machine learning model for the Iris Data set.

**Step1.** The program starts by reading first training dataset and creates structures of input feautures and known outputs.
> Inputs  
> ------  
> Matrix(104 rows,4 columns)  
> 5.1,3.5,1.4,0.2  
> 4.9,3,1.4,0.2  
> 4.7,3.2,1.3,0.2  
> 4.6,3.1,1.5,0.2  
> 5,3.6,1.4,0.2  
> 5.4,3.9,1.7,0.4  
> ...  
> ...  
> 6.4,2.8,5.6,2.2  
> 6.3,2.8,5.1,1.5  
> 6.1,2.6,5.6,1.4  

> Outputs  
> --------  
> 1,0,0  
> 1,0,0  
> 1,0,0  
> 1,0,0  
> 1,0,0  
> 1,0,0  
> ...  
> ...  
> 0,0,1  
> 0,0,1  
> 0,0,1  

**Step2.** We define the architecture of the neural network for this particular problem. Our neural network concists of 3 layers, input, hidden and output layer.

![Original image](/docs/dnn.png "Neural Network Architecture")

**Step 3.** We generate randomly the initial weights for our newral network and for that 2 matrices, with their dimensnions relying on the number of feautures in the hidden layer and the number labels of the output layer. 

> Initial Weights  
> -----------------------------------------------------   
> Matrix(4 rows,10 columns)   
> -0.392277,0.432058,0.326602,-0.473844,0.970544,0.459071,0.850452,0.183974,-1.16253,-1.10889   
> -0.225372,-1.35322,1.43599,0.376165,0.277973,0.59594,-1.21472,-1.23385,0.432877,0.620929   
> -1.53982,1.39396,1.89087,1.6739,-1.02046,0.497299,1.73487,0.6594,-0.787059,-0.15031  
> -1.40425,-1.6524,-1.88987,-0.73973,0.119533,1.84754,0.442807,0.724385,-0.7012,-0.357806   

> Matrix(10 rows,3 columns)  
> -0.392277,0.432058,0.326602   
> -0.473844,0.970544,0.459071  
> 0.850452,0.183974,-1.16253  
> -1.10889,-0.225372,-1.35322  
> 1.43599,0.376165,0.277973  
> 0.59594,-1.21472,-1.23385  
> 0.432877,0.620929,-1.53982  
> 1.39396,1.89087,1.6739  
> -1.02046,0.497299,1.73487   
> 0.6594,-0.787059,-0.15031  

**Step 4.** We train the model using feedforward and backpropagation to produce the final weights. 
As we see in the output below,the accuracy is increasing while the error decreases.

> Model Training...  
> -----------------------------------------------------  
> epochs:1============================= accuracy:74.7598 loss:26.2498  
> epochs:2============================= accuracy:82.2554 loss:18.4543  
> epochs:3============================= accuracy:85.8427 loss:14.7236  
> epochs:4============================= accuracy:86.7974 loss:13.7307  
> epochs:5============================= accuracy:87.2016 loss:13.3104  
> epochs:6============================= accuracy:87.4976 loss:13.0025  
> epochs:7============================= accuracy:87.7466 loss:12.7436  
> ...  
> ...  
> ...  
> epochs:47============================= accuracy:94.0113 loss:6.2282  
> epochs:48============================= accuracy:94.2059 loss:6.02583  
> epochs:49============================= accuracy:94.395 loss:5.82921  
> epochs:50============================= accuracy:94.5782 loss:5.63864  

> Final Weights  
> -----------------------------------------------------  
> Matrix(4 rows,10 columns)  
> -0.434039,0.165075,0.328048,-1.12876,1.14849,0.524303,0.901431,-0.27736,-1.22554,-0.485483  
> -0.253925,-2.02443,1.43697,-0.315379,0.692018,0.641787,-1.17897,-1.50819,0.389557,1.48529  
> -1.55183,2.39843,1.89126,2.08749,-2.05135,0.513446,1.74975,1.18252,-0.805449,-0.796153  
> -1.40598,-1.14365,-1.88984,-0.423301,-0.498645,1.84895,0.445474,1.13896,-0.703917,-0.673524  
> Matrix(10 rows,3 columns)  
> -0.383105,0.422995,0.323035  
> -2.85654,1.69145,1.65192  
> 0.0759394,-0.439495,-0.890426  
> -2.93016,-1.19215,0.414916  
> 2.32713,1.3129,-1.55974  
> -0.185406,-1.83234,-0.958555  
> -0.441043,0.0971356,-1.22873  
> -0.164877,0.3483,3.63923  
> -1.01455,0.491782,1.73242  
> 1.22277,-1.26229,-0.535522  


**Step 5.**
> Model Testing on Test Samples  
> -----------------------------------------------------  
> Input/Output  data where imported successfully...  

> Actual Y value:Iris-setosa Predicted Y value:Iris-setosa Accuracy: 92.7785  
> Actual Y value:Iris-setosa Predicted Y value:Iris-setosa Accuracy: 93.262  
> Actual Y value:Iris-setosa Predicted Y value:Iris-setosa Accuracy: 90.1079  
> Actual Y value:Iris-setosa Predicted Y value:Iris-setosa Accuracy: 91.3809  
> Actual Y value:Iris-setosa Predicted Y value:Iris-setosa Accuracy: 92.2176  
> ...  
> ...  
> Actual Y value:Iris-virginica Predicted Y value:Iris-virginica Accuracy: 90.7138  
> Actual Y value:Iris-virginica Predicted Y value:Iris-virginica Accuracy: 88.4969  
> Actual Y value:Iris-virginica Predicted Y value:Iris-virginica Accuracy: 89.5119  
> Actual Y value:Iris-virginica Predicted Y value:Iris-virginica Accuracy: 87.9169  
> Actual Y value:Iris-virginica Predicted Y value:Iris-virginica Accuracy: 88.9618  
> Actual Y value:Iris-virginica Predicted Y value:Iris-virginica Accuracy: 87.9063  


## Code structure 
```Matrix``` struct object : A data structure that contains information about matrix dimensions,   
             the dynamic memory that holds the vectorized data of the matrix, and all the functions and overloaded   
             operators needed for algebraic matrix operations.  
```dataset```: A class definition and implementation that serves the functionality related to handling training and testing data.  
```modelclasses``` : A singleton class object for managing information about labels, which are the outputs of the implemented neural network.  
             The rest of the code is organized into .cu c++ code files and their corresponding header files. In most cases, each file implements one component 
             of the newral network, keeping the code readable and easier to maintain.  
             Finally, in the the main.cu, we define the neural network architecture, train the model,
             and show the results of the trained model by predicting the outputs according to the inputs from the test dataset.  