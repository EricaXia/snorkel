# Extracting Virus-Host relations from Academic Papers - Machine Learning Project Using Snorkel

In this project, we build a Snorkel application to extract and identify correct mentions of virus and animal host species from academic research papers. The tasks are broken up into each step of the pipeline. 

[Part 1](snorkel_part_1.ipynb)
**- Document Preparation, Preprocessing, and Candidate Extraction**
- Read in a corpus of documents in .tsv format
- Extract candidates through dictionary matching

[Part 2](snorkel_part_2.ipynb)
**- Labeling Functions Development**
- Develop Labeling Functions to label candidates as true or false
- Compare LF performance with hand labeled set (gold labels)

[Part 3](snorkel_part_3.ipynb)
**- Generative Model Training**
- Unify the LFs and reduce their noise
- Marginal predictions from the model are used for the probabilistic training labels for the end extraction model

[Part 4](snorkel_part_4.ipynb)
**- LSTM Neural Network Training**
- Train a LSTM model using training labels from Part 3
- Evaluate model performance on a blind test set

### Additional Data
- [Metadata table](https://github.com/EricaXia/snorkel/blob/master/metadata.tsv) 


