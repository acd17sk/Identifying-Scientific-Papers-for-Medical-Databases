# LitCovid Research Paper Classifier 🔬

This project automates the process of identifying relevant COVID-19 research papers for the LitCovid database. By leveraging text classification, it filters articles from PubMed, significantly reducing manual screening time for curators and researchers.

> **Problem:** With the rapid online publication of research, administrators of specialized databases like LitCovid spend countless hours manually screening articles.
>
> **Solution:** This project implements a machine learning pipeline to automatically classify research papers, creating a list of articles suitable for inclusion in the LitCovid database. It compares various classifiers and feature extraction techniques to determine the most effective combination.

---

## 🚀 Features

-   **Data Retrieval**: Fetches the latest research articles directly from PubMed.
-   **Text Pre-processing**: Cleans and prepares raw article text for model training.
-   **Multiple Models**: Implements and compares several classification models:
    -   Naive Bayes (NB)
    -   Support Vector Machine (SVM)
    -   Logistic Regression (LR)
    -   Convolutional Neural Network (CNN)
-   **Flexible Feature Extraction**: Utilizes both **TF-IDF** and **Word2Vec** for text vectorization.
-   **Automated Prediction**: Generates a clean list of relevant article PMIDs based on the user's chosen configuration.

---

## 📁 Project Structure
```
## 📁 Project Structure

.
├── Predictions/
│   └── nb_tfidf_daily_balanced_Pos_Predictions.txt   # Example output file
├── metrics/
│   └── ...                                           # Stores trial results
├── 03312021.litcovid.export.tsv                      # LitCovid PMIDs for training labels
├── all_data_ret.py                                   # Retrieves last 100,000 articles
├── daily_data_ret.py                                 # Retrieves articles from a specific date
├── pre-processing.py                                 # Cleans and processes text data
├── predictor.py                                      # Main script to train and predict
├── CNN.py                                            # CNN Class definition
├── Cnn_model.py                                      # CNN model architecture
└── W2v_class.py                                      # Word2Vec Class helper

```
