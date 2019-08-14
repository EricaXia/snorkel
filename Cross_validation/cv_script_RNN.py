# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 11:29:51 2019

@author: ericaxia3
"""
# Try RNN end extrator

# Import all libraries
import sqlite3
import numpy as np
import pandas as pd
import re
from snorkel import SnorkelSession
from snorkel.parser.spacy_parser import Spacy
from snorkel.parser import CorpusParser
from snorkel.models import Document, Sentence
from snorkel.parser import TSVDocPreprocessor
from snorkel.matchers import DictionaryMatch
from snorkel.candidates import Ngrams, CandidateExtractor
from snorkel.models import candidate_subclass
from snorkel.lf_helpers import (
    get_left_tokens, 
    get_right_tokens, 
    get_between_tokens,
    get_text_between, 
    get_tagged_text,
    rule_regex_search_tagged_text,
    rule_regex_search_btw_AB,
    rule_regex_search_btw_BA,
    rule_regex_search_before_A,
    rule_regex_search_before_B,
)
import bz2
from snorkel.annotations import LabelAnnotator
from sklearn.model_selection import train_test_split
from snorkel.annotations import load_gold_labels
from util_virushost import load_external_labels
from snorkel.learning import GenerativeModel
from snorkel.learning.structure import DependencySelector
from features import hybrid_span_mention_ftrs
from sklearn.metrics import roc_curve, auc, precision_recall_curve, precision_score, recall_score, accuracy_score
from matplotlib import pyplot as plt
from scipy import interp

plt.rcParams["figure.figsize"] = [12,8]

#------------------------------------------
## Define some useful sql functions
def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by the db_file
    :param db_file: database file
    :return: Connection object or None
    """
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)
 
    return None

def delete_all(conn):
    """
    Delete all rows in the table
    :param conn: Connection to the SQLite database
    :return:
    """
    sql = 'DELETE FROM virus_host' # change this to the table to delete
    cur = conn.cursor()
    cur.execute(sql)

# main function to delete sql tables
def main():
    database = "snorkel.db"
    # create a database connection
    conn = create_connection(database)
    with conn:
        delete_all(conn);
#-------------------------------------
domestic_names = pd.read_csv('domestic_names.csv')
names_list = domestic_names.iloc[:,0].tolist()
ictv_animals = pd.read_csv('ictv_animals.csv')
ictv_series = ictv_animals.stack().reset_index().iloc[:,2]
ictv_list = ictv_series.tolist()

def name(s): 
    # split the string into a list  
    l = s.split() 
    new_word = ""  # begins as empty string
    if len(l) == 2:
        for i in range(len(l)-1): 
            s = l[i] 
            # adds the capital first character  
            new_word += (s[0].upper()+'. ') 
        new_word += l[-1].title() # add the last word
        return new_word 
    else:
        return s
    

ictv_list2 = [name(s) for s in ictv_list] # shortened species names list
animals_list = list(set(names_list + ictv_list + ictv_list2))
# Create a list of virus names
ictv_viruses = pd.read_csv('ictv_viruses.csv')
# create copies of certain virus names without the digit at the end
ictv_viruses['Species2'] = ictv_viruses['Species'].str.replace('\d+', '', regex=True)
ictv_v_series = ictv_viruses.stack().reset_index().iloc[:,2].drop_duplicates()
virus_list = ictv_v_series.tolist()
virus_abbrev = pd.read_csv('virus_abbrev.csv', header = None)
virus_list = virus_list + virus_abbrev.iloc[:,0].tolist() 
# Clean up white space and remove any empty strings
animals_list = [animal.strip() for animal in animals_list]
animals_list = list(filter(None, animals_list))
virus_list = [virus.strip() for virus in virus_list]
virus_list = list(filter(None, virus_list))
# remove terms we don't want to match
dont_want = ['Mal', 'mal', 'Ou', 'ou', 'Marta', 'marta', 'León', 'león', 'Euro', 'euro']

for word in dont_want:
    if word in animals_list:
        animals_list.remove(word)

dont_want2 = ['bat', 'BAT', 'langur', 'mcp', 'MCP', 'con', 'CON', 'spf', 'SPF', '(SPF)', 'his', 'His', 'HIS', 'pfu', 'PFU', '(PFU)', '(NSP)', 'mal', 'MAL', 'Mal', 'ifa', 'IFA', '(IFA)', 'wrc', 'WRC', '(WRC)', 'fitc', '(fitc)', 'fam', 'Cherry', 'cherry', 'ihc', 'IHC', '(IHC)', 'hit'] 

for word in dont_want2:
    if word in virus_list:
        virus_list.remove(word)
        
# ------------------------------------------

# START SNORKEL SESSION

session = SnorkelSession()

n_docs = 500

doc_preprocessor = TSVDocPreprocessor('pdfs_big.tsv', max_docs=n_docs) # new files (88 papers)
corpus_parser = CorpusParser(parser=Spacy())
corpus_parser.apply(doc_preprocessor, count=n_docs)

VirusHost = candidate_subclass('VirusHost', ['virus', 'host'])

ngrams = Ngrams(n_max=10)
virus_matcher = DictionaryMatch(d = virus_list)
animals_matcher = DictionaryMatch(d = animals_list)
cand_extractor = CandidateExtractor(VirusHost, [ngrams, ngrams], [virus_matcher, animals_matcher], nested_relations = True)

docs = session.query(Document).order_by(Document.name).all()

# Text Pattern based labeling functions, which look for certain keywords

# List to parenthetical
def ltp(x):
    return '(' + '|'.join(x) + ')'


# --------------------------------

# Positive LFs:

detect = {'detect', 'detects', 'detected', 'detecting', 'detection', 'detectable'}
detect_l = ['detect', 'detects', 'detected', 'detecting', 'detection', 'detectable']
infect = {'infect', 'infects', 'infected', 'infecting', 'infection'}
isolate = {'isolate', 'isolates', 'isolated', 'isolating', 'isolation'}
other_verbs = {
    'transmit(ted)?', 'found', 'find(ings)?', 'affect(s|ed|ing)?', 'confirm(s|ed|ing)?', 'relat(ed|es|e|ing|ion)?', 'recovered', 'identified', 'collected'
}
misc = {
    'seropositive', 'seropositivity', 'positive', 'host(s)?', 'prevalen(ce|t)?', 'case(s)?', 'ELISA', 'titer', 'viremia', 'antibod(y|ies)?', 'antigen', 'exposure', 'PCR', 'polymerase chain reaction', 'RNA', 'DNA', 'nucleotide', 'sequence', 'evidence', 'common', 'success(fully)?', 'extract(ed)?', 'PFU', '(PFU)', 'plaque-forming unit', 'suscept', 'probably', 'probable', 'high(er)?'
}

causal = ['caus(es|ed|e|ing|ation)?', 'induc(es|ed|e|ing)?', 'associat(ed|ing|es|e|ion)?']

positive = {'detect', 'detects', 'detected', 'detecting', 'detection', 'detectable', 'infect', 'infects', 'infected', 'infecting', 'infection', 'isolate', 'isolates', 'isolated', 'isolating', 'isolation'}
positive_l = ['detect', 'detects', 'detected', 'detecting', 'detection', 'detectable', 'infect', 'infects', 'infected', 'infecting', 'infection', 'isolate', 'isolates', 'isolated', 'isolating', 'isolation']

# negative words
negative = {
    'negative (antibodies)?', 'seronegative', 'seronegativity', 'negate', 'not', 'Not', '\bno\b', '\bNo\b', '(titer(s)?\W+(?:\w+\W+){1,6}?less than)', 'titers against', 'none', 'resist', 'never', 'unlikely'
}
negative_l = [
    'negative (antibodies)?', 'seronegative', 'seronegativity', 'negate', 'not', 'Not', '\bno\b', '\bNo\b', '(titer(s)?\W+(?:\w+\W+){1,6}?less than)', 'titers against', 'none', 'resist', 'never', 'unlikely'
]
neg_rgx = r'|'.join(negative_l)

# search nearby words for negatives, returns True if negative word found:
def neg_nearby(c):  
    if (len(negative.intersection(get_between_tokens(c))) > 0):
        return True
    elif (len(negative.intersection(get_left_tokens(c, window=15))) > 0):
        return True
    else:
        return False


# words like detect 
def LF_detect(c):
    if (len(detect.intersection(get_between_tokens(c))) > 0) and not neg_nearby(c):
        return 1
    elif (len(detect.intersection(get_left_tokens(c[0], window=20))) > 0) and not neg_nearby(c):
        return 1
    elif (len(detect.intersection(get_left_tokens(c[1], window=20))) > 0) and not neg_nearby(c):
        return 1
    elif (len(detect.intersection(get_right_tokens(c[0], window=20))) > 0) and not neg_nearby(c):
        return 1
    elif (len(detect.intersection(get_right_tokens(c[1], window=20))) > 0) and not neg_nearby(c):
        return 1
    else:
        return 0
    
def LF_infect(c):
    if len(infect.intersection(get_between_tokens(c))) > 0  and not neg_nearby(c):
        return 1
    elif len(infect.intersection(get_left_tokens(c[0], window=20))) > 0 and not neg_nearby(c):
        return 1
    elif len(infect.intersection(get_left_tokens(c[1], window=20))) > 0 and not neg_nearby(c):
        return 1
    elif len(infect.intersection(get_right_tokens(c[0], window=20))) > 0 and not neg_nearby(c):
        return 1
    elif len(infect.intersection(get_right_tokens(c[1], window=20))) > 0 and not neg_nearby(c):
        return 1
    else:
        return 0
    
    # Words like 'isolated'
def LF_isolate(c):
    if len(isolate.intersection(get_between_tokens(c))) > 0 and not neg_nearby(c):
        return 1
    elif len(isolate.intersection(get_left_tokens(c[0], window=20))) > 0 and not neg_nearby(c):
        return 1
    elif len(isolate.intersection(get_left_tokens(c[1], window=20))) > 0 and not neg_nearby(c):
        return 1
    elif len(isolate.intersection(get_right_tokens(c[0], window=20))) > 0 and not neg_nearby(c):
        return 1
    elif len(isolate.intersection(get_right_tokens(c[1], window=20))) > 0 and not neg_nearby(c):
        return 1
    else:
        return 0

        
def LF_misc(c):
    if (len(misc.intersection(get_between_tokens(c))) > 0) and not neg_nearby(c):
        return 1
    elif (len(misc.intersection(get_left_tokens(c[0], window=20))) > 0) and not neg_nearby(c):
        return 1
    elif (len(misc.intersection(get_left_tokens(c[1], window=20))) > 0) and not neg_nearby(c):
        return 1
    elif (len(misc.intersection(get_right_tokens(c[0], window=20))) > 0) and not neg_nearby(c):
        return 1
    elif (len(misc.intersection(get_right_tokens(c[1], window=20))) > 0) and not neg_nearby(c):
        return 1
    else:
        return 0
    
# terms like 'virus A caused disease in host B'
def LF_v_cause_h(c):
    return 1 if (
        re.search(r'{{A}}.{0,50} ' + ltp(causal) + '.{0,50}{{B}}', get_tagged_text(c), re.I)
        and not re.search('{{A}}.{0,50}(not|no|negative).{0,20}' + ltp(causal) + '.{0,50}{{B}}', get_tagged_text(c), re.I)
    ) else 0

# if candidates are nearby and check for negative words
def LF_v_h(c):
    return 1 if (
        re.search(r'{{A}}.{0,200}{{B}}', get_tagged_text(c), re.I)
        and not re.search(neg_rgx, get_tagged_text(c), re.I)
    ) else 0

def LF_h_v(c):
    return 1 if (
        re.search(r'{{B}}.{0,250}{{A}}', get_tagged_text(c), re.I)
        and not re.search(neg_rgx, get_tagged_text(c), re.I)
    ) else 0

# positive verbs (detect, infect, isolate)
def LF_positive(c):
    if (len(positive.intersection(get_between_tokens(c))) > 0) and not neg_nearby(c):
        return 1
    elif (len(positive.intersection(get_left_tokens(c[0], window=20))) > 0) and not neg_nearby(c):
        return 1
    elif (len(positive.intersection(get_left_tokens(c[1], window=20))) > 0) and not neg_nearby(c):
        return 1
    elif (len(positive.intersection(get_right_tokens(c[0], window=20))) > 0) and not neg_nearby(c):
        return 1
    elif (len(positive.intersection(get_right_tokens(c[1], window=20))) > 0) and not neg_nearby(c):
        return 1
    else:
        return 0
    
def LF_positive2(c):
    return 1 if (
        re.search(r'{{A}}.{0,100} ' + ltp(positive_l) + '.{0,100}{{B}}', get_tagged_text(c), re.I)
        and not re.search('{{A}}.{0,100}(not|no|negative).{0,20}' + ltp(positive_l) + '.{0,100}{{B}}', get_tagged_text(c), re.I)
    ) else 0

def LF_other_verbs(c):
    if (len(other_verbs.intersection(get_between_tokens(c))) > 0) and not neg_nearby(c):
        return 1
    elif (len(other_verbs.intersection(get_left_tokens(c[0], window=20))) > 0) and not neg_nearby(c):
        return 1
    elif (len(other_verbs.intersection(get_left_tokens(c[1], window=20))) > 0) and not neg_nearby(c):
        return 1
    elif (len(other_verbs.intersection(get_right_tokens(c[0], window=20))) > 0) and not neg_nearby(c):
        return 1
    elif (len(other_verbs.intersection(get_right_tokens(c[1], window=20))) > 0) and not neg_nearby(c):
        return 1
    else:
        return 0

# -----------------------------------

# Negative LFs:

# if candidate pair is too far apart, mark as negative
def LF_far_v_h(c):
    return rule_regex_search_btw_AB(c, '.{350,5000}', -1)

def LF_far_h_v(c):
    return rule_regex_search_btw_BA(c, '.{350,5000}', -1)

def LF_neg_h(c):
    return -1 if re.search(neg_rgx + '.{0,50}{{B}}', get_tagged_text(c), flags=re.I) else 0

def LF_neg_assertions(c):
    if (len(negative.intersection(get_between_tokens(c))) > 0): 
        return -1
    elif (len(negative.intersection(get_left_tokens(c[0], window=10))) > 0):
        return -1
    elif (len(negative.intersection(get_left_tokens(c[1], window=20))) > 0):
        return -1
    elif (len(negative.intersection(get_right_tokens(c[0], window=20))) > 0):
        return -1
#    elif (len(negative.intersection(get_right_tokens(c[1], window=20))) > 0):
#        return -1
    else:
        return 0
    
# Distant Supervision LFs
# Compare candidates with a database of known virus-host pairs (from Virus-Host Database)
# Function to remove special characters from text
def strip_special(s):
    return ''.join(c for c in s if ord(c) < 128)

# Read in known pairs and save as set of tuples
with bz2.BZ2File('virushostdb.tar.bz2', 'rb') as f:
    known_pairs = set(
        tuple(strip_special(x.decode('utf-8')).strip().split('\t')) for x in f.readlines()
    )

def LF_distant_supervision(c):
    v, h = c.virus.get_span(), c.host.get_span()
    return 1 if (v,h) in known_pairs else 0

# list of all LFs
LFs = [
     LF_detect, LF_infect, LF_isolate, LF_positive, LF_positive2, LF_misc, LF_v_cause_h, LF_v_h, LF_h_v, LF_other_verbs, LF_far_v_h, LF_far_h_v, LF_neg_h, LF_neg_assertions, LF_distant_supervision
]

# set up the label annotator class
labeler = LabelAnnotator(lfs=LFs)

# -------------------------------------------

# START CROSS VALIDATION SPLIT in a loop:

# Make an array of indexes (should equal number of documents 88). In a loop, split the index array into   train, test, and dev arrays. The sentences get added to the respective t,t,d sets and the candidates are extracted.
        

index_array = np.arange(0, 88)

# for roc
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

# for recording prec, rec, f1 scores
precs = []
recalls = []
f1s = []
accs = []

# Trying 3 fold cross validation
for fold in range(5):
    
    print("starting fold {}".format(fold+1))
    main() # clears the virus_host sql table

    X_train, X_test = train_test_split(index_array, test_size=0.2, random_state=None)
    X_train, X_dev = train_test_split(X_train, test_size=0.2, random_state=None)

    train_sents = set()
    dev_sents   = set()
    test_sents  = set()
    for i, doc in enumerate(docs):
        for s in doc.sentences:
            if i in X_dev:
                dev_sents.add(s)
            elif i in X_test:
                test_sents.add(s)
            else:
                train_sents.add(s)
    # Number of sentences per set
    print("Sentences per train, dev, and test sets:", 
       len(train_sents), len(dev_sents), len(test_sents))
    
    # Candidate extraction and define as train, dev, and test
    for i, sents in enumerate([train_sents, dev_sents, test_sents]): 
        cand_extractor.apply(sents, split=i)
        
    train_cands = session.query(VirusHost).filter(VirusHost.split == 
                            0).order_by(VirusHost.id).all()
    dev_cands   = session.query(VirusHost).filter(VirusHost.split == 
                            1).order_by(VirusHost.id).all()
    test_cands  = session.query(VirusHost).filter(VirusHost.split == 
                            2).order_by(VirusHost.id).all()

    # Apply labeler to all sets
    L_train = labeler.apply(split=0)
    L_dev = labeler.apply(split=1)
    L_test = labeler.apply(split=2)

    # Load gold labels
    missed = load_external_labels(session, VirusHost, annotator_name = 'gold', split = 1)
    L_gold_dev = load_gold_labels(session, annotator_name='gold', split = 1)
    missed = load_external_labels(session, VirusHost, annotator_name = 'gold', split = 2)
    L_gold_test = load_gold_labels(session, annotator_name = 'gold', split = 2)
    
    
    # Generative model
    ds = DependencySelector()
    deps = ds.select(L_train, threshold = 0.1)
    
    gen_model = GenerativeModel()
    gen_model.train(L_train, epochs=100, decay=0.95, step_size=0.1 / L_train.shape[0], reg_param=1.00e-03, deps=deps)
    
    train_marginals = gen_model.marginals(L_train)
    

    # Discriminative model
    from snorkel.learning.tensorflow.rnn import reRNN

    rnn = reRNN()
    rnn.train(X_train=train_cands, Y_train=train_marginals, X_dev=dev_cands, Y_dev=L_gold_dev, n_epochs=30)

    p, r, f1 = rnn.score(X_test = test_cands, Y_test = L_gold_test, set_unlabeled_as_neg=False)
    print("Prec: {0:.3f}, Recall: {1:.3f}, F1 Score: {2:.3f}".format(p, r, f1))

    precs.append(p) # save for later
    recalls.append(r)
    f1s.append(f1)

    tp, fp, tn, fn = rnn.error_analysis(session, test_cands, L_gold_test, set_unlabeled_as_neg = False)

    accuracy = (len(tp)+len(tn))/(len(tp)+len(tn)+len(fp)+len(tn))
    accs.append(accuracy)    

    # save marginals (raw predictions on candidates)
    test_marginals = rnn.marginals(X = test_cands)

   # =============================================================================  
    # calculate prc and roc curves
    
    gold_nums = []
    for gold in L_gold_test:
        new_i = gold.todense()
        gold_nums.append(new_i[0,0])
    y = np.asarray(gold_nums)
    
     # Prec-Recall Curve (in loop)
    y_df = pd.DataFrame(y)
    probs_df = pd.DataFrame(test_marginals)
    new_df = pd.concat([y_df, probs_df], axis=1)
    new_df.columns = 0,1
    new_df = new_df[new_df[0] != 0]
    new_y = new_df[0].to_numpy()
    new_test_marginals = new_df[1].to_numpy()
     
     # calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(new_y, new_test_marginals, pos_label=1)
     
     # plot the current precision-recall curve
    plt.plot(recall, precision, marker='.', label='Fold %d' % (fold+1))
 
    
 # AFTER LOOP
 # plot no skill
plt.plot([0, 1], [0.5, 0.5], linestyle='--')
 # show final plot
plt.title("5-Fold CV Precision Recall Curve")
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc="lower right")
plt.savefig("5_fold_cv_prc_RNN.png")
plt.show()

    
print(precs)
print(recalls)
print(f1s)
print(accs)

df = pd.DataFrame(list(zip(precs, recalls, f1s, accs)), 
                  index =['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5'],
                  columns =["Precision", "Recall", "F1", "Accuracy"]) 
df2 = df.round(2)

# output to Latex Table for the current model used
print(df2.to_latex(index=True))



# plt.plot(df2)
# leg = plt.legend(df2.columns)

# =============================================================================
# dcsummary = pd.DataFrame([df2.mean()],index=['Mean'])
# # get the individual lines inside legend and set line width
# for line in leg.get_lines():
#     line.set_linewidth(5)
# plt.table(cellText=dcsummary.values,colWidths = [0.25]*len(df2.columns),
#           rowLabels=dcsummary.index,
#           colLabels=dcsummary.columns,
#           cellLoc = 'center', rowLoc = 'center',
#           loc='top')
# =============================================================================
# fig = plt.gcf()
# plt.table(cellText=df2.values,colWidths = [0.25]*len(df2.columns),
#           rowLabels=df2.index,
#           colLabels=df2.columns,
#           cellLoc = 'center', rowLoc = 'center',
#           loc='top'    )
# print('\n')
# plt.savefig('cross_validation_trend.png', bbox_inches = "tight")
# plt.show()


# =============================================================================
# # marginals distr plot (of the last fold)
# plt.figure(figsize=(12,8))
# plt.hist(train_marginals, bins=20, range=(0.0, 1.0))
# plt.title('Distribution of Training Marginals')
# plt.savefig('train_marginals_final.png')
# plt.show()
# 
# =============================================================================










