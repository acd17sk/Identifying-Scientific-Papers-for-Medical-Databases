Abstract:

Living in a digital world, research studies are being published online at a higher rate than ever before. This growing number of published studies is creating a huge problem for administrators to regulate them, thus databases are becoming incredibly difficult to maintain. Administrators such as researchers, have to spend countless of hours to screen and identify particular published research papers that fits into the category of their interest. LitCovid is a curated literature hub that is explicitly used for the purpose of collecting data that are specifically related to COVID-19 and SARS-CoV-2. The purpose of this project is to filter out published research papers that are not relevant to the LitCovid database. A list has to be created of papers that are identified to be suitable to be included in the LitCovid database by adopting text classification procedures. This will drastically reduce the amount of time that is spent by the administrators screening research papers to identify which are relevant and thus enabling them to use their valuable time in more important tasks. This project will conclude by comparing various feature extraction techniques combined with various classifiers and determine which brings the best results. Classifiers include, Naive Bayes, SVM, Logistic Regression and Convolutional Neural Network.


Folders & Files explanation and Instructions:

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Folder “metrics” –  where trial results were stored

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
03312021.litcovid.export.tsv  -- Litcovid pmids
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Processed data:


++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Pubmed retrieval files:


daily_data_ret.py – retrieves articles uploaded on a specific date
SAVES RESULTS AS “unprocessed_{all/daily}_data.pkl
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Pre-processing:


TO USE IN COMMAND LINE “python pre-processing -m “{daily/all}”

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



TO USE IN COMMAND LINE
SAVES RESULTS OF POSITIVE CLASSED PMIDS IN FOLDER ”PREDICTIONS” BASED ON CONFIGURATION SELECTED BY THE USER
“{nb/cnn/lr/svm}_{tfidf/w2v}_{daily/all}_{balanced/imbalanced}_Pos_Predictions.txt”

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
CNN class:



++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++