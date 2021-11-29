from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from w2v_class import W2V
import torch
from torch.utils.data import TensorDataset
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from CNN import CNN
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import numpy as np
import sys, getopt


class CommandLine:
    def __init__(self):
        opts, args = getopt.getopt(sys.argv[1:], 'd:c:v:m:')
        opts = dict(opts)
        self.exit = True

        if '-m' in opts:
            if opts['-m'] in ('all', 'daily'):
                self.mode = opts['-m']
                self.exit = False

            else:
                warning = (
                    "*** ERROR: MODE label (opt: -m LABEL)! ***\n"
                    "    -- value (%s) not recognised!\n"
                    "    -- must be one of: all / daily"
                    )  % (opts['-m'])
                print(warning, file=sys.stderr)
                self.exit = True
                return
        else:
            self.mode = 'daily'
            self.exit = False

        if '-c' in opts:
            if opts['-c'] in ('nb', 'cnn', 'lr', 'svm'):
                self.classifier = opts['-c']
                self.exit = False

            else:
                warning = (
                    "*** ERROR: MODE label (opt: -c LABEL)! ***\n"
                    "    -- value (%s) not recognised!\n"
                    "    -- must be one of: nb / cnn / lr / svm"
                    )  % (opts['-c'])
                print(warning, file=sys.stderr)
                self.exit = True
                return
        else:
            warning = (
                "*** ERROR: MODE label (opt: -c LABEL)! ***\n"
                "    -- must be one of: nb / cnn / lr / svm"
                )
            print(warning, file=sys.stderr)
            self.exit = True
            return

        if '-v' in opts:
            if opts['-v'] in ('tfidf' , 'w2v'):
                self.vectorization_mode = opts['-v']
                self.exit = False

            else:
                warning = (
                    "*** ERROR: MODE label (opt: -v LABEL)! ***\n"
                    "    -- value (%s) not recognised!\n"
                    "    -- must be one of: tfidf / w2v"
                    )  % (opts['-v'])
                print(warning, file=sys.stderr)
                self.exit = True
                return
        else:
            warning = (
                "*** ERROR: MODE label (opt: -v LABEL)! ***\n"
                "    -- must be one of: tfidf / w2v"
                )  % (opts['-v'])
            print(warning, file=sys.stderr)
            self.exit = True
            return

        if '-d' in opts:
            if opts['-d'] in ('balanced', 'imbalanced'):
                self.balanced_mode = opts['-d']
                self.exit = False

            else:
                warning = (
                    "*** ERROR: MODE label (opt: -m LABEL)! ***\n"
                    "    -- value (%s) not recognised!\n"
                    "    -- must be one of: balanced / imbalanced"
                    )  % (opts['-d'])
                print(warning, file=sys.stderr)
                self.exit = True
                return
        else:
            self.balanced_mode = 'imbalanced'
            self.exit = False


"""
data_spliter method splits data into training data - labels
and testing data - labels
** if -m is 'all' then its going to split all the data to 70% training
30% testing
** if -m is daily then the training data will be all data contained and
testing data will bel a day's data
returns: training data - labels
         testing data - labels
         dataframe of testing data that is in same order as testing data
"""
def data_spliter(mode):

    if mode == 'all':
        from math import ceil

        data = pd.read_pickle('processed_all_data.pkl')
        data = data.sample(frac=1).reset_index()


        positive_data = data[data['LABEL'] == 1]
        negative_data =  data[data['LABEL'] == 0]

        pos_data_length = len(positive_data)
        neg_data_length = len(negative_data)

        tr_pos_len = ceil(pos_data_length*0.7)
        tr_neg_len = ceil(neg_data_length*0.7)

        pos_training_pd = positive_data.iloc[:tr_pos_len]
        neg_training_pd = negative_data.iloc[:tr_neg_len]
        pos_testing_pd = positive_data.iloc[tr_pos_len:]
        neg_testing_pd = negative_data.iloc[tr_neg_len:]

        training_pd = pos_training_pd.append(neg_training_pd)
        testing_pd = pos_testing_pd.append(neg_testing_pd)

        x_train = np.array(list(training_pd['process_AB']))

        x_test = np.array(list(testing_pd['process_AB']))

        y_train = np.array(list(training_pd['LABEL']))

        y_test = np.array(list(testing_pd['LABEL']))


    else: # daily
        data = pd.read_pickle('processed_all_data.pkl')

        testing_pd = pd.read_pickle('processed_daily_data.pkl')

        x_train = np.array(list(data['process_AB']))

        x_test = np.array(list(testing_pd['process_AB']))

        y_train = np.array(list(data['LABEL']))

        y_test = np.array(list(testing_pd['LABEL']))

    print("SPLIT COMPLETE")

    return x_train, x_test, y_train, y_test, testing_pd

"""
data_vectorizer vectorizes the dataset
** if -v is 'tfidf' then the vectorization method that is going to be used
is tf-idf
** if -v is 'w2v' then the vectorization method that is going to be used
is word2vec
returns: vectorized training and testing data
"""
def data_vectorizer(vec_mode, x_train, x_test):

    if vec_mode == 'tfidf':
        tfidf = TfidfVectorizer(ngram_range = (1,2),
                                lowercase=False,
                                norm='l2',
                                max_df = 0.7,
                                min_df = 0.01,
                                max_features=250,
                                sublinear_tf=True,
                                smooth_idf=True)

        fitted_vectorizer = tfidf.fit([" ".join(text) for text in x_train])

        print("TFIDF TRAIN COMPLETE")

        vec_x_train = fitted_vectorizer.transform([" ".join(text) for text in x_train]).toarray()

        vec_x_test = fitted_vectorizer.transform([" ".join(text) for text in x_test]).toarray()

        print("TFIDF VECTORIZATION COMPLETE")
    else: # w2v
        w2v = W2V(mode=1,n_w_features=50, n_features=50)

        w2v_vectorizer = w2v.fit(x_train)

        print("W2V TRAIN COMPLETE")

        vec_x_train = w2v_vectorizer.transform(x_train)

        vec_x_test = w2v_vectorizer.transform(x_test)

        print("W2V VECTORIZATION COMPLETE")

    return vec_x_train, vec_x_test

"""
data_balancer uses smote in order to balance the number of
records for the categories in the trining dataset
returns: resampled data - labels
"""
def data_balancer(vec_x_train, y_train):
    sm = SMOTE()
    X_res, y_res = sm.fit_resample(vec_x_train, y_train)

    return X_res, y_res

"""*****************************************************************************

***** nb_classifier & cnn_classifier & lr_classifier & svm_classifier ********
nb_classifier = Gaussian Naive Bayes
cnn_classifier = Convolutional Neural Network
lr_classifier = Logistic Regression
svm_classifier = Support Vector Machine
the following methods contain the classification models
according to the specifications of the user input from the terminal
the parameters will vary
returns predicted labels - accuracy
"""
def nb_classifier(vec_mode, balanced_mode, x_train, x_test, y_train, y_test):
    if vec_mode == 'tfidf':
        mm = GaussianNB(var_smoothing=8.111308307896872e-07)

    else: # w2v
        if balanced_mode == 'balanced':
            mm = GaussianNB()
        else: # imbalanced
            mm = GaussianNB(var_smoothing=1.0)

    mm.fit(x_train, y_train)
    pp = mm.predict(x_test)
    accuracy = accuracy_score(y_test, pp)

    return pp, accuracy

def cnn_classifier(vec_mode, x_train, x_test, y_train, y_test):

    y_trainl = np.zeros((y_train.shape[0], 2))
    y_testl = np.zeros((y_test.shape[0], 2))

    for i in range(0,y_train.shape[0]):
        y_trainl[i, y_train[i].astype(int)]=1
    for i in range(0,y_test.shape[0]):
        y_testl[i, y_test[i].astype(int)]=1


    x_tr, x_val, y_tr, y_val = train_test_split(x_train,
                                                y_trainl,
                                                test_size=0.2,
                                                shuffle=True)

    td_tr = TensorDataset(torch.tensor(x_tr, dtype=torch.float32), torch.tensor(y_tr))
    td_val = TensorDataset(torch.tensor(x_val, dtype=torch.float32), torch.tensor(y_val))
    td_te = TensorDataset(torch.tensor(x_test, dtype=torch.float32), torch.tensor(y_testl))

    if vec_mode == 'tfidf':
        mm = CNN()
    else: # w2v
        mm = CNN(num_features = 2500, stride_conv=25, epochs=15)

    mm.fit(td_tr,td_val)
    pp, accuracy = mm.predict(td_te)

    return pp, accuracy

def lr_classifier(vec_mode, x_train, x_test, y_train, y_test):
    if vec_mode == 'tfidf':
        mm = LogisticRegression(penalty='l1', max_iter=100, C=0.1, solver='liblinear')
    else: # w2v
        mm = LogisticRegression(penalty='l1', max_iter=1000, C=0.1, solver='liblinear')

    mm.fit(x_train, y_train)
    pp = mm.predict(x_test)
    accuracy = accuracy_score(y_test, pp)

    return pp, accuracy

def svm_classifier(vec_mode, x_train, x_test, y_train, y_test):
    if vec_mode == 'tfidf':
        mm = LinearSVC(penalty='l1', max_iter=2500, C=0.1, dual=False)
    else: # w2v
        mm = LinearSVC(penalty='l1', max_iter=1000, C=0.01, dual=False)

    mm.fit(x_train, y_train)
    pp = mm.predict(x_test)
    accuracy = accuracy_score(y_test, pp)

    return pp, accuracy

if __name__ == '__main__':
    config = CommandLine()
    if config.exit:
        sys.exit(0)

    mode = config.mode
    balanced_mode = config.balanced_mode
    vec_mode = config.vectorization_mode
    classifier = config.classifier


    x_train, x_test, y_train, y_test, testing_pd = data_spliter(mode)

    vec_x_train, vec_x_test = data_vectorizer(vec_mode, x_train, x_test)

    if balanced_mode == 'balanced':
        c_input, c_labels = data_balancer(vec_x_train, y_train)
    else: # imbalanced
        c_input, c_labels = vec_x_train, y_train


    if classifier == 'nb':
        predicted, accuracy = nb_classifier(vec_mode,
                                            balanced_mode,
                                            c_input,
                                            vec_x_test,
                                            c_labels,
                                            y_test)
    elif classifier == 'lr':
        predicted, accuracy = lr_classifier(vec_mode,
                                            c_input,
                                            vec_x_test,
                                            c_labels,
                                            y_test)
    elif classifier == 'cnn':
        predicted, accuracy = cnn_classifier(vec_mode,
                                             c_input,
                                             vec_x_test,
                                             c_labels,
                                             y_test)
    else: # svm
        predicted, accuracy = svm_classifier(vec_mode,
                                             c_input,
                                             vec_x_test,
                                             c_labels,
                                             y_test)

    testing_pd['Predictions'] = predicted
    positive_predictions = testing_pd[testing_pd['Predictions'] == 1]
    filename = f'./Predicted/{classifier}_{vec_mode}_{mode}_{balanced_mode}_Pos_Predictions.txt'

    output_series = pd.Series(positive_predictions['PMID'])
    output_series.to_csv(filename, header = ['PMID'], index=False)

    print('Accuracy Score:', accuracy)
    print(f'Predictions are stored in: "{filename}"')
