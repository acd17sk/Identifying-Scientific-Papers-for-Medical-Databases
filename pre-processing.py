import pandas as pd
import numpy as np
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import sys, getopt

import string

class CommandLine:
    def __init__(self):
        opts, args = getopt.getopt(sys.argv[1:], ':m:')
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

def remove_sections_citations(dataset):
    d = dataset.str.replace('[A-Z &]+[:]', ' ', case=True)
    d = d.str.replace('[\[][0-9]+[\]]', ' ')
    return d

def lower_case_text(dataset):
    d = dataset.str.lower()
    return d

def remove_punctuation(dataset):
    # punctuation characters
    # !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    d = ["".join([char for char in text if char not in string.punctuation]) for text in dataset]
    return d

def tokenize_text(dataset):
    d = [word_tokenize(text) for text in dataset]
    return d

def remove_stop_words(dataset):
    stop_words = stopwords.words('english')
    d = [[word for word in text if word not in stop_words] for text in dataset]
    return d

def stem_text(dataset):
    porter = PorterStemmer()
    d = [[porter.stem(word) for word in text] for text in dataset]
    return d

# pd.set_option('display.max_colwidth', None)
if __name__ == '__main__':
    config = CommandLine()
    if config.exit:
        sys.exit(0)
    mode = config.mode

    data_decider_dict = {'all': {0:'unprocessed_all_data.pkl',
                                 1: 'processed_all_data.pkl'},
                         'daily': {0:'unprocessed_daily_data.pkl',
                                   1: 'processed_daily_data.pkl'}}

    data = pd.read_pickle(data_decider_dict[mode][0])


    data['process_AB'] = remove_sections_citations(data['AB'])
    data['process_AB'] = lower_case_text(data['process_AB'])
    data['process_AB'] = remove_punctuation(data['process_AB'])
    data['process_AB'] = tokenize_text(data['process_AB'])
    data['process_AB'] = remove_stop_words(data['process_AB'])
    data['process_AB'] = stem_text(data['process_AB'])


    data.to_pickle(data_decider_dict[mode][1])
