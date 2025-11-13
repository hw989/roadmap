# Deep Learning Lab 1: Study of Deep Learning Packages

## 📚 What This Lab Is About
This lab introduces you to the four major deep learning frameworks/packages that are commonly used in artificial intelligence and machine learning projects. Think of these as different toolboxes that help you build smart programs that can learn from data.

---

## 🎯 The Four Main Packages

### 1. **TensorFlow**
**What is it?**
- TensorFlow is like a powerful engine developed by Google that helps computers learn patterns from data
- It's one of the most popular tools for building AI models
- The name comes from "tensors" (multi-dimensional arrays of numbers) that "flow" through the network

**What happens in the code:**
```python
!pip install tensorflow
```
- This command downloads and installs TensorFlow on your computer
- The `!pip install` is like going to an app store and downloading an app
- After installation, you can use all of TensorFlow's features to build neural networks

**Key Features:**
- Works well with large datasets
- Can run on both CPUs (regular processors) and GPUs (graphics cards that make training faster)
- Has excellent support for deploying models to production
- Backed by Google, so it's constantly updated and maintained

---

### 2. **Keras**
**What is it?**
- Keras is like a simplified, user-friendly interface that sits on top of TensorFlow
- Think of TensorFlow as a car's engine, and Keras as the simple dashboard that makes it easy to drive
- It makes building neural networks much easier with less code

**What happens in the code:**
```python
from keras import datasets
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
```

**Breaking it down:**
1. **`from keras import datasets`**: We're importing the datasets module from Keras
2. **`mnist.load_data()`**: Loading the MNIST dataset, which contains 70,000 images of handwritten digits (0-9)
3. **`train_images, train_labels`**: 60,000 images for training the model (teaching it)
4. **`test_images, test_labels`**: 10,000 images for testing how well the model learned

**Checking the shapes:**
```python
train_images.shape  # Shows (60000, 28, 28) - 60,000 images of 28×28 pixels
test_images.shape   # Shows (10000, 28, 28) - 10,000 images of 28×28 pixels
```

**Key Features:**
- Super easy to use - great for beginners
- Can build complex neural networks with just a few lines of code
- Now officially part of TensorFlow (as tf.keras)
- Has many pre-built layers and models you can use

---

### 3. **Theano**
**What is it?**
- Theano is one of the older deep learning libraries
- It was developed at the University of Montreal
- While it's no longer actively developed, it influenced many modern frameworks

**What happens in the code:**
```python
!pip install Theano
```
- This installs Theano on your system
- The output shows it's already installed with its dependencies:
  - **numpy**: For handling arrays and mathematical operations
  - **scipy**: For scientific computing
  - **six**: For Python 2 and 3 compatibility

**Key Features:**
- Was one of the first deep learning frameworks
- Very efficient at mathematical computations
- Helped pave the way for TensorFlow and PyTorch
- Note: Development stopped in 2017, but it's still useful for learning concepts

---

### 4. **PyTorch**
**What is it?**
- PyTorch is Facebook's (Meta's) answer to TensorFlow
- It's become extremely popular, especially in research
- Known for being more "Pythonic" (feels more natural to Python programmers)

**What happens in the code:**
```python
!pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu115
```

**Breaking down the installation:**
- **torch**: The main PyTorch library
- **torchvision**: Tools for working with images
- **torchaudio**: Tools for working with audio data
- **cu115**: CUDA 11.5 support (for running on NVIDIA GPUs)

**Using PyTorch:**
```python
import torch
import torch.nn as nn
print(torch.__version__)  # Shows which version is installed (2.1.0+cpu)
torch.cuda.is_available()  # Checks if GPU is available for faster training
```

**Key Features:**
- Very intuitive and easy to debug
- Excellent for research and experimentation
- Dynamic computation graphs (more flexible than TensorFlow's static graphs)
- Strong community support
- Great documentation and tutorials

---

## 🔍 Key Concepts Explained

### What is a Neural Network?
Imagine teaching a child to recognize animals:
- You show them pictures of dogs
- They learn features: four legs, tail, fur, ears
- Eventually, they can identify new dogs they've never seen

Neural networks work similarly:
- You show them data (images, text, numbers)
- They learn patterns and features
- They can then make predictions on new, unseen data

### What are Tensors?
- Think of tensors as containers for data:
  - **0D Tensor** (Scalar): A single number, like 5
  - **1D Tensor** (Vector): A list of numbers, like [1, 2, 3]
  - **2D Tensor** (Matrix): A table of numbers, like an image
  - **3D Tensor**: Multiple matrices stacked together
  - **Higher dimensions**: For more complex data

### CPU vs GPU
- **CPU** (Central Processing Unit): Your computer's main brain, good at general tasks
- **GPU** (Graphics Processing Unit): Originally for graphics, but excellent for deep learning because it can do many calculations in parallel
- Training on GPU can be 10-100x faster than CPU!

---

## 🎓 Summary

This lab gives you an overview of the main tools available for deep learning:

1. **TensorFlow**: Industry standard, powerful, backed by Google
2. **Keras**: Beginner-friendly, now part of TensorFlow
3. **Theano**: Historical importance, influenced modern frameworks
4. **PyTorch**: Research favorite, intuitive, backed by Facebook/Meta

**Which one should you use?**
- **Starting out?** Use Keras (with TensorFlow backend)
- **Doing research?** PyTorch is very popular
- **Building production systems?** TensorFlow has excellent deployment tools
- **Learning the fundamentals?** Any of them will work!

The good news is that the core concepts are similar across all frameworks, so learning one makes it easier to pick up others.

---

## 💡 Real-World Applications

These frameworks are used to build:
- **Image Recognition**: Face detection in phones, medical image analysis
- **Natural Language Processing**: Chatbots, language translation, sentiment analysis
- **Recommendation Systems**: Netflix suggestions, YouTube recommendations
- **Autonomous Vehicles**: Self-driving cars
- **Voice Assistants**: Siri, Alexa, Google Assistant
- **Game AI**: Chess engines, video game opponents

---

## 🚀 Next Steps

After understanding these packages, you'll typically:
1. Choose a framework (usually TensorFlow/Keras or PyTorch)
2. Learn to load and prepare data
3. Build your first neural network
4. Train the model on data
5. Evaluate its performance
6. Make predictions on new data

Remember: The best way to learn is by doing! Don't worry if you don't understand everything at once - deep learning is a journey, and each project teaches you something new.

