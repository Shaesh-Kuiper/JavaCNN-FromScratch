# JavaCNN – A Pure Java CNN for FashionMNIST

This project demonstrates a complete implementation of a Convolutional Neural Network (CNN) in Java from scratch, without any external libraries. It includes the full training and inference pipeline for the FashionMNIST dataset, highlighting core deep learning algorithms and data structures.

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

   
