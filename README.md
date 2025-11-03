Chinese News Text Analysis with BERT
A natural language processing project for Chinese news text analysis, featuring document clustering, classification, and semantic search using BERT embeddings. Developed during a machine learning research internship.
Overview
This project implements a comprehensive text analysis pipeline for Chinese news articles:

Document Clustering: Unsupervised grouping using KMeans on BERT embeddings
Text Classification: Fine-tuned BERT model for multi-class classification
Semantic Search: Cosine similarity-based document retrieval
Dimensionality Reduction: PCA visualization of high-dimensional embeddings
Translation: Chinese-to-English text translation using MarianMT

Dataset

Source: THUCNews (Tsinghua University Chinese News Corpus)
Category: Sports News (体育)
Size: 16,896 documents
Language: Chinese

Technical Stack

Deep Learning: PyTorch, Transformers (Hugging Face)
Models:

BERT (bert-base-uncased) for text embeddings
BertForSequenceClassification for classification
MarianMT (Helsinki-NLP) for translation


ML Libraries: Scikit-learn (KMeans, PCA, StandardScaler)
Visualization: Matplotlib

Repository Structure
├── BertClassification&Clustering.ipynb    # Clustering, classification, search
├── BertGeneration.ipynb                   # Text translation (Chinese to English)
└── README.md

## Features

### 1. Text Vectorization
- Converts Chinese text to 768-dimensional BERT embeddings
- Standardizes vectors using StandardScaler
- Handles batch processing with GPU acceleration

### 2. Clustering Analysis
- Implements KMeans clustering (k=5)
- Evaluates optimal cluster number using elbow method
- Calculates silhouette scores for cluster quality
- PCA visualization (2D) for cluster inspection

### 3. Classification
- Fine-tunes BERT for 5-class classification
- Training: 3 epochs with AdamW optimizer (lr=2e-5)
- Performance: 97% accuracy on sports news dataset
- Batch size: 16, Max sequence length: 100

### 4. Semantic Search
- Query-based document retrieval using cosine similarity
- Returns most similar document to user query
- Supports both English and Chinese queries

### 5. Translation
- Chinese-to-English translation using MarianMT
- Preserves paragraph and sentence structure
- GPU-accelerated inference
- Processes documents with Chinese period (。) as delimiter
