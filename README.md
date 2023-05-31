# Medical Specialty Classification
In this repository, we utilized BERT models to develop a a classification model in classifying transcription texts to medical specialties. We implemented all models using PyTorch version 3.9, where the Intel 8th generation CPU performed all simulation in this study witn an NVIDIA RTX 1050Ti 4GB graphics card.

You can access the main Jupyter Notebook in the file:
```
../src/RADI623_A1_RomenSamuelWabina.ipynb
```

## Dependencies
You need to install the prerequisites.
```
pip install -r requirements.txt
```

## Background

The problem of predicting one’s illnesses wrongly through self-diagnosis in medicine is very real. In a report by the [Telegraph](https://www.telegraph.co.uk/news/health/news/11760658/One-in-four-self-diagnose-on-the-internet-instead-of-visiting-the-doctor.html), nearly one in four self-diagnose instead of visiting the doctor. Out of those who misdiagnose, nearly half have misdiagnosed their illness wrongly [reported](https://bigthink.com/health/self-diagnosis/). While there could be multiple root causes to this problem, this could stem from a general unwillingness and inability to seek professional help.

Elevent percent of the respondents surveyed, for example, could not find an appointment in time. This means that crucial time is lost during the screening phase of a medical treatment, and early diagnosis which could have resulted in illnesses treated earlier was not achieved.

With the knowledge of which medical specialty area to focus on, a patient can receive targeted help much faster through consulting specialist doctors. To alleviate waiting times and predict which area of medical specialty to focus on, we can utilize natural language processing (NLP) to solve this task.

Given any medical transcript or patient condition, this solution would predict the medical specialty that the patient should seek help in. Ideally, given a sufficiently comprehensive transcript (and dataset), one would be able to predict exactly which illness he is suffering from.


<center>
<img src = "/figures/framework_final.png" width = "808"/>
</center>

The vocabulary used in the field of medicine is extremely specialized, unique, and frequently made up of complicated jargon that is only applicable to that field. Therefore, using a specialized tokenizer created especially for medical texts can have a number of advantages in terms of precision, comprehension, and context preservation. Many medical phrases contain compound words, acronyms, abbreviations, and unique symbols that may not be properly handled by general tokenizers. 

BlueBERT and RoBERTa are some of the tokenizers which can handle medical texts. However, for the sake of learning, we want to train our own tokenizer using the given dataset as part of the proposed model to determine if it can improved our classification performance. We can construct our own tokenizer to successfully handle such scenarios by using the given dataset <code>mtsamples</code>. Becuase of this, better tokenization of medical terminology is made possible by this level of personalization, avoiding the loss of important information during preprocessing. 

<center>
<img src = "/figures/medical_tokenizer.PNG" width = "808"/>
</center>

We start creating the tokenizer by instantiating a <code>Tokenizer</code> object with a model, then set its <code>normalizer, pre_tokenizer, post_processor</code>, and <code>decoder</code> attribute to the values we want. We specified the <code>[UNK]</code> token so that the model knows what to return when it encounters characters it hasn't seen before. We utilized the WordPiece tokenizer as the foundation of our customized tokenizer. During tokenization, the first step is normalization. Since BERT is widely used, there is a BertNormalizer with the classic options we can set for BERT: lowercase and strip_accents; clean_text to remove all control characters and replace repeating spaces with a single one.

|   Model        | Baseline LSTM (50) | BaselineLSTM (128) | BaselineLSTM (256) |
|----------------|--------------------|--------------------|--------------------|
|   BERT         |  59.376            | 55.468             | 47.264             |
|   RoBERTa      |  66.073            | 56.211             | 43.313             |
|   DistilBERT   |  73.438            | 56.099             | 54.282             |
|   BlueBERT     |  71.070            | 67.160             | 58.594             |

