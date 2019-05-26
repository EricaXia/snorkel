# Extracting Virus-Host Relations from Research Text - Machine Learning Project Using Snorkel

![pic](logo.png)

In this project, we build a machine learning system to extract and identify correct mentions of virus and animal host species relations from academic research papers in the context of epidemiological research.

<details><summary>Read summary here</summary>
<p>
  
  Considering a large majority of infectious diseases are spread from animals to humans, zoonotic diseases have become an important topic of study and the subject of many research studies. Various species of viruses, such as Flaviviruses, may cause the outbreak of viral zoonotic disease. Hence, the relations between viral and animal host species are major factors in understanding the transmission and characteristics of zoonotic diseases. Natural Language Processing extraction techniques can be used to identify species-level mentions of viral-host relations in academic text. 

  In this project, we build a system to extract and identify correct mentions of virus and animal host species from academic research papers. The goal of such methods is to provide insights into the scientific writing and international research conducted on species linked to zoonotic disease. After extracting frequencies of the mentions of specific viral-host relations, we use supervised machine learning techniques to label entity pairs as having positive or negative associations. 

  One challenge in the way of applying supervised learning methods is the creation of large, labeled training sets. In our project, we require training sets of confirmed viral and host species relations. Hence, we use data programming by way of a training set creation package called Snorkel (created by HazyResearch from Stanford Dawn project) to create training set. The training sets are noisy, machine labeled sets created by applying user-defined heuristics, called labeling functions, to extracted candidate pairs. A generative model is deployed to unify the labeling functions and reduce noise in the final training set. Finally, end extraction is performed by an LSTM model to predict correct relation mentions.    
</p>
</details>


## Code

The tasks are broken up into each step of the pipeline. 


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
- Use marginal predictions from the model as the probabilistic training labels (for the end extraction model in Part 4)

[Part 4](snorkel_part_4.ipynb)
**- LSTM Neural Network Training**
- Train a LSTM model using training labels from Part 3
- Evaluate model performance on a blind test set

### Additional Data
- [Metadata table](https://github.com/EricaXia/snorkel/blob/master/metadata.tsv) 


