from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import *

import pandas as pd
from snorkel.models import StableLabel
from snorkel.db_helpers import reload_annotator_labels

FPATH = r'external_labels_current.txt'

def load_external_labels(session, candidate_class, split, annotator_name='gold'):
    gold_labels = pd.read_csv(FPATH, sep="\t")
    
    # Get split candidates
    candidates = session.query(candidate_class).filter(
        candidate_class.split == split
    ).all()
    
    for index, row in gold_labels.iterrows():    

        # We check if the label already exists, in case this cell was already executed
        context_stable_ids = "~~".join([row['virus'], row['host']])
        query = session.query(StableLabel).filter(StableLabel.context_stable_ids == context_stable_ids)
        query = query.filter(StableLabel.annotator_name == annotator_name)
        
        # If label doesn't exist, add label to the session
        if query.count() == 0:
            session.add(StableLabel(
                context_stable_ids=context_stable_ids,
                annotator_name=annotator_name,
                value=row['label']))   
            
    # Commit session
    session.commit()

    # Reload annotator labels
    reload_annotator_labels(session, candidate_class, annotator_name, split=split, filter_label_split=False)
    