# JavaCNN – A Pure Java CNN for FashionMNIST

This project demonstrates a complete implementation of a Convolutional Neural Network (CNN) in Java from scratch, without any external libraries. It includes the full training and inference pipeline for the FashionMNIST dataset, highlighting core deep learning algorithms and data structures.

At the end of 5 epochs you can expect a accuracy of 87+ %

## 🚀 Run the Project

1. **Clone the repository**
   ```bash
   git clone git@github.com:Shaesh-Kuiper/JavaCNN-FromScratch.git
   cd JavaCNN-FromScratch

2. **Compile all .java files**
   ```bash
   javac -encoding UTF-8 *.java

3. **Run the training**
   ```bash
   java Trainer

  **NOTE**: Training time depends on the number of epochs (on average, 1 epoch takes 5–10 minutes).
            After training, a file named trained-model.bin will be saved containing the learned model.

4. **Run inference**
   ```bash
   java Inference trained-model.bin data\FashionMNIST\raw\t10k-images-idx3-ubyte data\FashionMNIST\raw\t10k-labels-idx1-ubyte

This runs prediction on a single sample from the test set.
It takes three arguments:
   1) Path to the trained model file
   2) Path to the test images file
   3) Path to the test labels file



   
## 🔍 Overview

**JavaCNN** is a learning project designed to understand how CNNs, gradient descent, and backpropagation work internally by building everything from the ground up. It covers:

- CNN architecture implementation  
- Manual backpropagation  
- Multithreaded execution  
- Data loading and processing  
- Training and evaluation loop  
- Model inference with softmax visualization  

---

## 🧠 Components

### 📦 `Conv2D.java`
Implements a convolutional layer:

- 3×3 kernels, stride 1, valid padding  
- Forward and backward passes  
- Xavier initialization  
- Parallelized with `ExecutorService` for output channels  

### 🌀 `MaxPool2D.java`
Implements 2×2 max pooling:

- Downsamples feature maps  
- Stores max positions for backprop  
- Sparse gradient propagation in backward pass  

### 🧮 `Dense.java`
Fully connected layer:

- Manual matrix multiplication  
- He initialization  
- Backprop with gradient update using SGD  

### 🔥 `ReLU.java`
Applies ReLU activation:

- Element-wise `max(0, x)`  
- Gradient zeroed for negatives during backward pass  

### 📉 `CrossEntropyLoss.java`
Handles loss calculation:

- Softmax + Negative log-likelihood  
- Stable computation via max-subtraction trick  
- Computes softmax gradient manually  

### 🧵 `ThreadPool.java`
Custom thread pool:

- 4-core fixed thread executor  
- Shared by layers like `Conv2D`  
- Optimizes CPU usage for forward/backward passes  

### 📁 `Utils.java`
Loads FashionMNIST data:

- Parses IDX format (images + labels)  
- Normalizes pixels to [0, 1]  
- Wraps data in a custom `DataSet` object  

### 🏗️ `CNNModel.java`
Defines the CNN architecture:

- 2 × (Conv → ReLU → Conv → ReLU → Pool)  
- Final dense layer for classification  
- Includes flattening from 3D to 1D  
- Supports full forward and backward training pass  

### 🏋️ `Trainer.java`
Trains the CNN:

- Loads and shuffles data  
- Trains over mini-batches (batch size = 64)  
- 2 training epochs using SGD  
- Evaluates accuracy on test set  
- Saves model to disk  

### 🧪 `Inference.java`
Performs image classification:

- Loads a sample image and the trained model  
- Runs forward pass and softmax  
- Displays image and predicted label in GUI  

---

## ⚙️ Algorithms & Data Structures

- Gradient descent and backpropagation implemented manually  
- Uses only native Java arrays (1D–4D)  
- Parallel execution for performance  
- Classic DSA usage: arrays, loops, task queues, producer-consumer threading  
- **No external ML or math libraries used**

---

## 📌 Highlights

- **End-to-end CNN implementation in Java**  
- **Educational focus**: understand the math and logic behind training neural networks  
- **Works entirely offline**  
- **Ideal for learners** exploring DL fundamentals  

---

## 🗂️ Dataset

- [FashionMNIST](https://github.com/zalandoresearch/fashion-mnist)  
- 28×28 grayscale images  
- 10 classes (e.g., sneakers, t-shirts, bags)  
- Data loading from IDX binary format  

---

## 🧠 Why?

This project is aimed at **demystifying deep learning** by writing everything yourself. It builds the bridge between **theoretical understanding** and **practical implementation** — completely in Java.

## 🙏 Usage & Attribution

This project is open for learning and inspiration. If you use or build upon this work, **please give credit** to the original author:

**Created by [Sarvesh R]**

GitHub: [Shaesh-Kuiper](https://github.com/Shaesh-Kuiper)

LinkedIn: [Sarvesh R](https://www.linkedin.com/in/sarveshrk/)

Email: [shvoyager2k4@gmail.com](shvoyager2k4@gmail.com)

A simple mention or link back is appreciated! 💙

