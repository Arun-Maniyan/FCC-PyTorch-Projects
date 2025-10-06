# Sarcasm Detection with PyTorch and Transformers

This project detects sarcasm in news headlines using deep learning and natural language processing techniques. It leverages PyTorch and state-of-the-art transformer models to classify headlines as sarcastic or not.

---

## Key Libraries Used

- **PyTorch**: Main framework for model building, training, and tensor operations.  
- **Transformers (HuggingFace)**: Provides `AutoTokenizer` and `AutoModel`, enabling the use of pre-trained BERT (`bert-base-uncased`) for embedding headlines.  
- **Pandas & NumPy**: For dataset loading, manipulation, and array operations.  
- **scikit-learn**: Used for train-test splits and evaluation metrics like accuracy.  
- **Matplotlib**: For plotting training and validation loss and accuracy curves.  

---

## Data Preprocessing

- **Dataset**: Headlines and their sarcasm annotations are loaded from `SarcasmHeadlinesDataset.json`.  
- **Cleaning**: Missing entries and duplicates are dropped. Unnecessary columns, such as article links, are removed.  
- **Splits**: Data is split into training, validation, and testing sets using `train_test_split` from scikit-learn.  
- **Tokenization**: Headlines are tokenized using `AutoTokenizer` from transformers, with padding, truncation, and maximum length limits. Tokenized sequences are converted into tensors suitable for BERT input.  

---

## Training Pipeline

### Custom Dataset Class
- Implements `torch.utils.data.Dataset` for efficient batching, tokenization, and device movement of headlines and labels.  

### DataLoader
- Utilizes `torch.utils.data.DataLoader` for mini-batching and shuffling across all splits.  

### Model Architecture
- **MyModel**: A custom class extending `nn.Module`, using a pre-trained BERT as a feature extractor (frozen during training).  
- The pooled BERT output passes through:
  - Two linear layers with ReLU activations  
  - Dropout for regularization  
  - Sigmoid activation for binary classification  

### Loss and Optimization
- **Loss Function**: Binary Cross-Entropy Loss (`nn.BCELoss`)  
- **Optimizer**: Adam  

### Training Loop
- Batches are passed through the model for each epoch.  
- Loss and accuracy are tracked for both training and validation sets.  
- Validation is performed after each epoch to monitor overfitting.  
- Final evaluation on the test set reports an accuracy of approximately **86%**.  

---

## Key Concepts

- **Transfer Learning**: Uses a pre-trained BERT model as a robust language feature extractor.  
- **Fine-Tuning**: While BERT is frozen, the final classifier head is trained on the sarcasm dataset, demonstrating partial model retraining.  
- **Tokenization & Embeddings**: Uses transformer-based contextual embeddings for better performance over traditional NLP pipelines.  
- **Efficient Batching & Device Management**: Ensures tensors and models are consistently moved to GPU if available.  
- **Model Evaluation**: Accuracy metrics and visualization of loss/accuracy curves across epochs help detect overfitting or underfitting.  

---

## Visualization
- Training and validation loss/accuracy curves are plotted using Matplotlib for monitoring model performance across epochs.  

---

## Usage

1. Install dependencies:

```bash
pip install torch transformers pandas scikit-learn matplotlib
