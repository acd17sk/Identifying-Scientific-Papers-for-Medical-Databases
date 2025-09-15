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


---

## 🛠️ How to Use

Follow these steps to run the full pipeline, from data retrieval to prediction.

### Step 1: Retrieve PubMed Data

First, retrieve the articles you want to classify. You have two options:

-   **All Data**: Get the last 100,000 articles published.
    ```bash
    python all_data_ret.py
    ```
    This will create `unprocessed_all_data.pkl`.

-   **Daily Data**: Get articles published on a specific date.
    ```bash
    python daily_data_ret.py
    ```
    This will create `unprocessed_daily_data.pkl`.

### Step 2: Pre-process the Data

Next, clean the raw text data. Run the pre-processing script, specifying the mode (`daily` or `all`).

```bash
# Example for daily data
python pre-processing.py -m daily

# Example for all data
python pre-processing.py -m all
```

This will generate the corresponding `processed_{daily/all}_data.pkl` file.

### Step 3: Train the Model and Predict
Finally, run the predictor to classify the articles. You need to specify the mode, classifier, vectorizer, and dataset balance.

Command Structure:
The main script is controlled with the following command and arguments:
```
python predictor.py -m {mode} -c {classifier} -v {vectorizer} -d {dataset}
```

- `{mode}`: daily or all

- `{classifier}`: nb, svm, lr, or cnn

- `{vectorizer}`: tfidf or w2v

- `{dataset}`: balanced or imbalanced

---

Example Command
To run a Support Vector Machine (SVM) with TF-IDF on the daily data using a balanced dataset, use the following command:

```
python predictor.py -m daily -c svm -v tfidf -d balanced
```
The results—a list of positively classified PMIDs—will be saved in the `Predictions/` folder. The output filename is based on the configuration, for example: `svm_tfidf_daily_balanced_Pos_Predictions.txt`.











