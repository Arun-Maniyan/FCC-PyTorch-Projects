# Audio Classification with PyTorch and CNNs

This project classifies audio samples (e.g., Quran recitations) using deep learning and state-of-the-art audio processing techniques. It leverages PyTorch and convolutional neural networks (CNNs) to assign class labels to audio files, such as speaker or content detection.

---

## Key Libraries Used

- PyTorch: Core framework for neural network construction, training, and tensor manipulations.  
- librosa: For audio loading and feature extraction (mel-spectrograms as CNN input).  
- Pandas & NumPy: For dataset handling and vectorized operations.  
- scikit-learn: Assists with label encoding and dataset splitting.  
- Matplotlib: Visualizes training and validation loss/accuracy curves.  

---

## Data Preprocessing

- Dataset: Audio file paths and class labels are stored in filespaths.csv.  
- Label Encoding: Class labels are converted to integers using LabelEncoder.  
- Splits: Data is split into training, validation, and test sets using random sampling.  
- Feature Extraction: Audio files are converted into mel-spectrograms via librosa, resized to 128×256, and scaled in decibels.  
- PyTorch Dataset: Custom dataset class efficiently loads and transforms spectrograms for batching.  

---

## Training Pipeline

### Custom Dataset Class
Implements torch.utils.data.Dataset to batch, transform, and serve spectrograms with labels.  

### DataLoader
Uses DataLoader for shuffling, mini-batching, and GPU device management across all splits.  

### Model Architecture
- Net: Custom CNN extending nn.Module with:
  - Three convolutional layers (ReLU + max pooling)  
  - Three fully connected layers and an output layer with dropout  
  - Softmax activation for multiclass prediction  

### Loss and Optimization
- Loss Function: Cross-Entropy Loss  
- Optimizer: Adam  

### Training Loop
- Batches are processed per epoch, logging training and validation loss and accuracy.  
- Early stopping and overfitting detection monitored via plotted curves.  
- Final test set evaluation yields accuracy of approximately 96.6%.  

---

## Key Concepts

- Audio Preprocessing & Augmentation: Ensures consistent spectrogram input size.  
- Custom CNN Architecture: Multi-layered convolutions capture spectral audio patterns.  
- Efficient Training & Device Management: Supports GPU acceleration.  
- Evaluation: Loss and accuracy logged per epoch; test set accuracy confirms performance.  

---

## Visualization
Training and validation loss/accuracy curves plotted with Matplotlib monitor performance and detect learning issues.  

---

## Usage

1. Install dependencies:  
pip install torch librosa pandas scikit-learn matplotlib

2. Organize audio data and filespaths.csv according to dataset structure.  

3. Run the notebook to preprocess data, train the model, and evaluate performance.
