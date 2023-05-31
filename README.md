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

## Process

The figure below illustrates a comprehensive pipeline for NLP text classification in the domain of medical transcription. This pipeline encompasses crucial steps necessary for achieving accurate classification results, including data preprocessing, tokenization, random oversampling, dataset iteration involving shuffle and batching, dataset splitting, embeddings, LSTM classification, and evaluation metrics. By providing a systematic and structured approach, this pipeline serves as a valuable guide for viewers, enabling them to effectively address the challenges specific to the medical transcription domain.

<center>
<img src = "/figures/framework_super.png" width = "808"/>
</center>

The vocabulary used in the field of medicine is extremely specialized, unique, and frequently made up of complicated jargon that is only applicable to that field. Therefore, using a specialized tokenizer created especially for medical texts can have a number of advantages in terms of precision, comprehension, and context preservation. Many medical phrases contain compound words, acronyms, abbreviations, and unique symbols that may not be properly handled by general tokenizers. 

BlueBERT and RoBERTa are some of the tokenizers which can handle medical texts. However, for the sake of learning, we want to train our own tokenizer using the given dataset as part of the proposed model to determine if it can improved our classification performance. We can construct our own tokenizer to successfully handle such scenarios by using the given dataset <code>mtsamples</code>. Becuase of this, better tokenization of medical terminology is made possible by this level of personalization, avoiding the loss of important information during preprocessing. 

<center>
<img src = "/figures/medical_tokenizer.PNG" width = "808"/>
</center>

We start creating the tokenizer by instantiating a <code>Tokenizer</code> object with a model, then set its <code>normalizer, pre_tokenizer, post_processor</code>, and <code>decoder</code> attribute to the values we want. We specified the <code>[UNK]</code> token so that the model knows what to return when it encounters characters it hasn't seen before. We utilized the WordPiece tokenizer as the foundation of our customized tokenizer. During tokenization, the first step is normalization. Since BERT is widely used, there is a BertNormalizer with the classic options we can set for BERT: lowercase and strip_accents; clean_text to remove all control characters and replace repeating spaces with a single one.

### Summary of pretrained BERT-based models without fine-tuning

|   Model        | Baseline LSTM (50) | BaselineLSTM (128) | BaselineLSTM (256) |
|----------------|--------------------|--------------------|--------------------|
|   BERT         |  59.376            | 55.468             | 47.264             |
|   RoBERTa      |  66.073            | 56.211             | 43.313             |
|   DistilBERT   |  73.438            | 56.099             | 54.282             |
|   BlueBERT     |  71.070            | 67.160             | 58.594             |

### Discussion: Proposed Model

We utilized the DistilBERT model with an LSTM layer, comprising 128 neurons, for the purpose of classifying medical specialties based on medical transcription text. To prepare the dataset, we employed our customized medical tokenizer, initialized using the BertTokenizerFast algorithm. Due to limited computational resources, we simulated the training process for only five epochs. The training was simulated using PyTorch version 3.9, with an Intel 8th generation CPU, and an NVIDIA RTX 1050Ti 4GB graphics card, similar to previous experiments. The results indicated potential signs of overfitting during training, as both the training and validation accuracies displayed a decreasing yet oscillating trend. To improve the training process, it would be beneficial to increase the number of epochs to an appropriate value once the model converges to the local (or global) minima. However, this would necessitate higher memory or GPU resources.

Recent studies have validated DistilBERT as a viable alternative to the original BERT model, particularly for handling medical text. For instance, <code>Abadeer (2020)</code> evaluated the performance of DistilBERT for the Named Entity Recognition (NER) task in medical records. This study aimed to determine how DistilBERT performs when fine-tuned with medical corpora compared to the pre-trained versions of BERT. The results demonstrated that DistilBERT achieved nearly identical performance to the medical versions of BERT in terms of F1-score. This implies that DistilBERT can provide excellent results, similar to BERT, but with reduced complexity, as the conventional BERT model contains 110 million parameters. However, a major limitation of DistilBERT is that it requires setting a maximum length of 512 tokens (similar to BERT), which proved insufficient for most sequences in the medical transcription dataset. Consequently, we had to split, truncate, and pad our tokens, potentially leading to highly sparse embedding vectors. Additionally, our findings align with <code>Abadeer (2020)'s</code> study, indicating that DistilBERT outperformed BlueBERT. <code>Abadeer (2020)</code> compared the two models and demonstrated that DistilBERT achieved comparable results with double the runtime speed. Thus, utilizing DistilBERT can offer a faster and more efficient model, particularly when fine-tuned, without sacrificing classification performance.

Most studies in natural language processing (NLP) focus on the pretrain-finetuning approach (PFA), which may present certain disadvantages, particularly for industries lacking sufficient server environments but requiring efficiency and high accuracy. Instead of solely searching for the best model to fulfill these objectives, researchers have developed their own tokenizers by examining the limitations of byte pair encoding (BPE) <code>(Sennrich et al., 2015)</code> and SentencePiece <code>(Kudo and Richardson, 2018)</code>. <code>Park et al. (2021)</code> adopted this approach and proposed an optimal tokenization method to improve machine translation performance based on morphological segmentation and vocabulary techniques. In our case, we trained a customized tokenizer with WordPiece as its foundation. Similarly, <code>Bilal et al. (2023)</code> employed a similar method to ours, loading a JSON file as an initial tokenizer for the BertTokenizerFast, which significantly enhanced their baseline models. The primary motivation behind creating such a tokenizer was the limited vocabulary of their language, Roman Urdu. Utilizing a specialized tokenizer designed specifically for medical texts provides several advantages in terms of precision, comprehension, and context preservation. Many medical phrases contain compound words, acronyms, abbreviations, and unique symbols that may not be properly handled by general tokenizers. Tokenizers like BlueBERT <code>(Peng et al., 2020)</code> and RoBERTa are capable of handling medical texts. However, for the purpose of learning, we opted to train our own tokenizer using the provided dataset, <code>mtsamples</code>, as part of the proposed model, to determine if it could improve our classification performance. This level of personalization enables better tokenization of medical terminology, avoiding the loss of important information during preprocessing.