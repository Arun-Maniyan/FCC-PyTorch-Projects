# Animal Face Recognition with PyTorch

This project builds a deep learning model for classifying animal faces (cats, dogs, and wild animals) using convolutional neural networks (CNNs) in PyTorch. It involves image classification on a dataset of labeled animal face images.

---

## Key Libraries Used

- PyTorch: Primary framework for model building, training, and tensor computations.  
- Torchvision: For image transformations such as resizing and tensor conversion.  
- Pandas & NumPy: Used for data loading and manipulation.  
- scikit-learn: Provides LabelEncoder for encoding categorical labels and functions for dataset splits.  
- Matplotlib: For visualization of training and validation metrics.  
- PIL (Pillow): For image loading and processing.  

---

## Data Preprocessing

- Dataset: Image paths and labels are gathered from structured folders containing cat, dog, and wild animal images.  
- Label Encoding: Animal categories are encoded to numerical labels (0, 1, 2) using LabelEncoder.  
- Splitting: Dataset divided into training (70%), validation, and testing sets using pandas sampling and index dropping.  
- Transformations: Images are resized to 128×128 pixels and converted to PyTorch tensors with appropriate float type.  
- Custom Dataset Class: torch.utils.data.Dataset subclass loads images on the fly, applies transformations, and returns image-label pairs for batching.  

---

## Training Pipeline

### Model Architecture
- Custom CNN with:
  - Three convolutional layers with increasing filter sizes (32, 64, 128), kernel size 3, padding 1  
  - Max Pooling layers after each convolution  
  - ReLU activations  
  - Flatten layer followed by two linear layers with 128 units and output layer for 3 classes  

### DataLoader
- Uses torch.utils.data.DataLoader for batching and shuffling during training and validation.  

### Loss and Optimization
- Loss Function: Cross-Entropy Loss for multi-class classification  
- Optimizer: Adam  

### Training Loop
- Each epoch:
  - Forward pass with training batches  
  - Loss calculation and backward propagation for weight updates  
  - Track and print training loss and accuracy  
  - Validation pass without gradients to monitor overfitting  
- Achieves around 99% validation accuracy by final epochs  

---

## Key Concepts

- Custom CNN Design: Tailored convolutional architecture for animal face classification  
- Data Splitting and Loader: Efficient preparation and real-time batch provisioning  
- GPU Acceleration: Model and tensors moved to GPU if available  
- Performance Tracking: Loss and accuracy plots visualize learning trends and diagnose issues  

---

## Visualization
- Matplotlib plots display training and validation loss and accuracy curves over epochs, providing insight into model convergence and generalization.  

---

## Usage

1. Install dependencies:  
pip install torch torchvision pandas scikit-learn matplotlib pillow

2. Organize dataset folder with labeled subfolders for each animal category.  

3. Run the notebook to preprocess data, define the model, and train the classifier.  

---
