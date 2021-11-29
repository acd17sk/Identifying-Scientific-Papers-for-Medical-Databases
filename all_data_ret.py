from Bio import Entrez, Medline
import pandas as pd
import numpy as np
import dateparser

def DP_to_DATE(dataset):
    dates = [dateparser.parse(d, date_formats=['%d/%m/%y']) for d in dataset]
    return dates

def create_labels(dataset):
    labels = []
    for d in dataset:
        labels.append(1 if int(d) in litcovid_pmids else 0)

    return labels

litcovid_pmids = set(list(pd.to_numeric(pd.read_csv('03312021.litcovid.export.tsv', sep='\t', header = 33)['pmid'])))


query = 'coronavirus[All Fields] OR ncov[All Fields] OR cov[All Fields] OR 2019-nCoV[All Fields] OR COVID-19[All Fields] OR SARS-CoV-2[All Fields]'
Entrez.email = "skonstantinou2@sheffield.ac.uk"
handle = Entrez.esearch(db="pubmed", retmax=100000, term=query)
record = Entrez.read(handle)
handle.close()


id_list_chunks = [record['IdList'][i:i + 10000] for i in range(0, len(record['IdList']), 10000)]

records_list = []

for ids in id_list_chunks:
    handle = Entrez.efetch(db="pubmed",id=[p for p in ids], rettype="medline", retmode="text")
    records = Medline.parse(handle)
    records_list.extend([r for r in records if 'AB' in r])


pubmed_data = pd.DataFrame(records_list, columns=['PMID', 'AB', 'DP'])
pubmed_data['DP'] = DP_to_DATE(pubmed_data['DP'])

pubmed_data['LABEL'] = create_labels(pubmed_data['PMID'])
number_of_included = pubmed_data[pubmed_data['LABEL'] == 1].shape[0]
number_not_included = pubmed_data[pubmed_data['LABEL'] == 0].shape[0]
print(number_of_included)
print(number_not_included)
perc_included = number_of_included*100 / pubmed_data.shape[0]
print(f"{round(perc_included,2)}%")

pubmed_data.to_pickle('./unprocessed_all_data.pkl')
print('DONE')
