# 📘 README.md  
## Learning 10-Dimensional Word Embeddings using CBOW and Skip-gram (Keras)

---

## 1. Overview

This project implements custom **10-dimensional word embeddings** from a domain-specific corpus using:
- CBOW (Continuous Bag of Words)
- Skip-gram

The entire pipeline is implemented strictly in **Python using Keras (TensorFlow backend)** without using any pre-trained embeddings.

---

## 2. Project Structure

├── 1dataset_builder.py  
├── 2train_cbow.py  
├── 3train_skipgram.py  
├── 4similarity.py  
├── README.md  
├── Output_Files/  
├── ScreenCaptures/  

---

## 3. Fixed Parameters

Embedding Dimension (D): 10  
Context Window Size (W): 4  
Default Learning Rates: 0.1, 0.01, 0.001  
Default Batch Size: 16  

---

## 4. Step-by-Step Execution

### Step 1: Dataset Creation
Run:
python dataset_builder.py

Outputs:
vocab.txt  
cbow_dataset.csv  
skipgram_dataset.csv  

---

### Step 2: CBOW Training
Run:
python train_cbow.py

- Uses one-hot encoding
- Performs learning-rate experiments
- Automatically selects best learning rate
- Saves final embeddings to embeddings_cbow.csv

---

### Step 3: Skip-gram Training
Run:
python train_skipgram.py

- Pair-wise target-context training
- Automatic learning-rate selection
- Saves final embeddings to embeddings_skipgram.csv

---

## 5. Learning-Rate Selection

Best learning rate is selected automatically based on:
- Loss convergence
- Stability of training (variance of last epochs)

---

## 6. Cosine Similarity Evaluation
Run:
python similarity.py

Features:
- Menu-driven interface
- Vocabulary displayed for selection
- Requires at least 5 query words
- Computes cosine similarity
- Displays Top-5 nearest neighbors
- Saves results to similarity_results.txt

---

## 7. Compliance

✔ Fixed W = 4  
✔ Fixed D = 10  
✔ No pre-trained embeddings  
✔ CBOW and Skip-gram implemented  
✔ Learning-rate experiments  
✔ Cosine similarity evaluation  

---

## 8. Requirements

Python 3.x  
TensorFlow / Keras  
NumPy  

---

## 9. Author

Roshan Sabu  
Word Embedding Learning
