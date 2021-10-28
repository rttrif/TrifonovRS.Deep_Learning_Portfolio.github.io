## PROJECT 24: SkimLit 

> ### TASK: Create an NLP model to classify abstract sentences into the role they play (e.g. objective, methods, results, etc) to enable researchers to skim through the literature (hence SkimLit) and dive deeper when necessary.


### Project goals and objectives

#### Project goal

- Studying **ANN model for sequential sentence classification**

#### Project objectives

1. Explore and prepare data 
2. Making a baseline (TF-IDF classifier)
3. Deep models with different combinations of: token embeddings, character embeddings, pretrained embeddings, positional embeddings
4. Building  multimodal model

### Dataset

[PubMed 200k RCT dataset](https://github.com/Franck-Dernoncourt/pubmed-rct)

**DATASET INFORMATION:**

PubMed 200k RCT is new dataset based on PubMed for sequential sentence classification. The dataset consists of approximately 200,000 abstracts of randomized controlled trials, totaling 2.3 million sentences. Each sentence of each abstract is labeled with their role in the abstract using one of the following classes: background, objective, method, result, or conclusion. The purpose of releasing this dataset is twofold. First, the majority of datasets for sequential short-text classification (i.e., classification of short texts that appear in sequences) are small: we hope that releasing a new large dataset will help develop more accurate algorithms for this task. Second, from an application perspective, researchers need better tools to efficiently skim through the literature. Automatically classifying each sentence in an abstract would help researchers read abstracts more efficiently, especially in fields where abstracts may be long, such as the medical field.

### Results

1. [ ] [**TF-IDF classifier**](https://github.com/rttrif/TrifonovRS.Deep_Learning_Portfolio.github.io/blob/main/Project%2024:%20SkimLit/SkimLit_TF_IDF_classifier.ipynb)
2. [ ] [**Conv1D with token embeddings**]()
3. [ ] [**Feature extraction with pretrained token embeddings**]()
4. [ ] [**Conv1D with character embeddings**]()
5. [ ] [**Combining pretrained token embeddings + character embeddings (hybrid embedding layer)**]()
6. [ ] [**Transfer Learning with pretrained token embeddings + character embeddings + positional embeddings**]()



### References

1. [PubMed 200k RCT: a Dataset for Sequential Sentence Classification in Medical Abstracts ](https://arxiv.org/pdf/1710.06071.pdf)
2. [Neural Networks for Joint Sentence Classification in Medical Paper Abstracts](https://arxiv.org/pdf/1612.05251.pdf)

