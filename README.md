# DNA Sequence Classification using 1D CNN & Transformer

A deep learning pipeline implemented in **PyTorch** for classifying DNA sequences into physiological states (Healthy, Irritable Bowel Syndrome, T2D Prediabetes). The project utilizes a hybrid neural network architecture combining **1D Convolutional Neural Networks (1D CNN)** for local motif extraction and **Transformer Encoders (Self-Attention)** to capture long-range genomic dependencies.

---

## 🚀 Key Features

* **K-Mer Tokenization:** Custom preprocessing pipeline that converts raw DNA sequences into overlapping $k$-mer tokens mapped via pre-trained biological Word2Vec embeddings.
* **Hybrid Architecture:** Seamless integration of PyTorch embedding layers, positional encoding, and `nn.TransformerEncoder` for capturing both local patterns and global context.
* **Algorithm Explainability:** Direct extraction and visualization of Multi-Head Attention weights to interpret how the model focuses on specific genomic motifs during classification.

---

## 📊 Model Performance & Metrics

The model was trained on a CUDA-enabled GPU and converged efficiently over 15 epochs. Evaluated on a test set of 1,256 sequences, the model achieved a **macro-average F1-score of 0.99**.

### Classification Report Analysis:
The model demonstrates exceptional calibration across all classes, prioritizing high recall for disease markers without sacrificing precision:
* **Healthy:** 1.00 Precision / 1.00 Recall.
* **Irritable Bowel Syndrome (IBS):** 0.97 Precision / 1.00 Recall (Zero false negatives).
* **T2D Prediabetes:** 1.00 Precision / 0.97 Recall.

```text
                          precision    recall  f1-score   support
                 Healthy       1.00      1.00      1.00       420
Irritable bowel syndrome       0.97      1.00      0.99       418
         T2D Prediabetes       1.00      0.97      0.98       418

                accuracy                           0.99      1256
```
🔍 Explainability: Opening the "Black Box"
To prove the model is learning actual biological patterns rather than dataset noise, we extract the Self-Attention Matrix from the final Transformer layer.

Above: A heatmap of the attention weights for a sample classified as 'Healthy'. The attention is correctly diffuse, indicating the absence of a dominant disease-specific motif, while still highlighting short-range k-mer interactions.

⚙️ Tech Stack
Language: Python 3.12

Frameworks: PyTorch (CUDA), Scikit-learn

NLP & Data Processing: Gensim (Word2Vec), NumPy, Pandas

Visualization: Matplotlib, Seaborn