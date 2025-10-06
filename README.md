# PyTorch Beginner Projects (FreeCodeCamp Tutorial)

This repository contains my implementations of the **5 beginner PyTorch projects** from the FreeCodeCamp YouTube tutorial.

I followed the tutorial closely but also made sure to understand each part of the code.

---

## Projects Included

1. **Rice Type Classification**  
   Classifies different types of rice using a feedforward neural network on tabular data.  
   - Preprocesses rice features from CSV, normalizes data, and splits into training, validation, and test sets.  
   - Uses a simple feedforward network with a hidden layer, trains with BCE loss, and evaluates accuracy.  

2. **Animal Face Recognition**  
   Classifies images of animal faces (cats, dogs, and wild animals) using a custom CNN.  
   - Prepares images with resizing and tensor conversion.  
   - Uses a 3-layer convolutional network with max pooling and fully connected layers.  
   - Achieves high validation accuracy (~99%) for multi-class classification.  

3. **Transfer Learning**  
   Applies transfer learning with GoogLeNet to classify bean leaf lesions.  
   - Loads images, applies resizing and transformations, and encodes labels.  
   - Replaces the final fully connected layer to match the dataset classes.  
   - Fine-tunes GoogLeNet and achieves validation accuracy around 92.5%.  

4. **Audio Classification**  
   Classifies audio samples (e.g., Quran recitations) using CNNs on mel-spectrograms.  
   - Extracts mel-spectrogram features from audio files.  
   - Trains a multi-layer CNN with convolutional and fully connected layers.  
   - Achieves test set accuracy of ~96.6%.  

5. **Sarcasm Detection**  
   Detects sarcasm in news headlines using a pre-trained BERT model.  
   - Tokenizes text with HuggingFace AutoTokenizer and uses BERT embeddings.  
   - Trains a classifier head on top of frozen BERT embeddings.  
   - Achieves test accuracy of ~86% and demonstrates transfer learning for NLP tasks.  

---

## Folder Structure

Each project has its own folder containing:  
- Jupyter Notebook (`.ipynb`) with code and explanations  
- Dataset files or links  
- Any additional scripts or utilities used in preprocessing or training  

---

## Acknowledgement

Based on the [Learn PyTorch with 5 Beginner Projects - FreeCodeCamp](https://www.youtube.com/watch?v=V_xro1bcAuA) tutorial.

