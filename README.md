# Medical Specialty Classification

The problem of predicting one’s illnesses wrongly through self-diagnosis in medicine is very real. In a report by the [Telegraph](https://www.telegraph.co.uk/news/health/news/11760658/One-in-four-self-diagnose-on-the-internet-instead-of-visiting-the-doctor.html), nearly one in four self-diagnose instead of visiting the doctor. Out of those who misdiagnose, nearly half have misdiagnosed their illness wrongly [reported](https://bigthink.com/health/self-diagnosis/). While there could be multiple root causes to this problem, this could stem from a general unwillingness and inability to seek professional help.

Elevent percent of the respondents surveyed, for example, could not find an appointment in time. This means that crucial time is lost during the screening phase of a medical treatment, and early diagnosis which could have resulted in illnesses treated earlier was not achieved.

With the knowledge of which medical specialty area to focus on, a patient can receive targeted help much faster through consulting specialist doctors. To alleviate waiting times and predict which area of medical specialty to focus on, we can utilize natural language processing (NLP) to solve this task.

Given any medical transcript or patient condition, this solution would predict the medical specialty that the patient should seek help in. Ideally, given a sufficiently comprehensive transcript (and dataset), one would be able to predict exactly which illness he is suffering from.




| Tokenizer               |  Embeddings    | BaselineLSTM (50) | BaselineLSTM (128)| BaselineLSTM (256) |
|:-----------------------:|----------------|-------------------|-------------------|--------------------|
| BERT                    |   BERT         |  59.376           | 55.468            | 47.264             |
| BERT                    |   RoBERTa      |  66.073           | 56.211            | 43.313             |
| BERT                    |   DistilBERT   |  73.438           | 56.099            | 54.282             |
| BERT                    |   ALBERT       |  71.070           | 67.160            | 58.594             |
| BERT                    |   BlueBERT     |  82%              | 89%               | 50%                |
| MedicalTokenizer + BERT |   BERT         |  84%              | 90%               | 61%                |
| MedicalTokenizer + BERT |   RoBERTa      |  50%              | 65%               | 15%                |
| MedicalTokenizer + BERT |   DistilBERT   |  50%              | 65%               | 15%                |
| MedicalTokenizer + BERT |   ALBERT       |  17%              | 49%               | 14%                |
| MedicalTokenizer + BERT |   BlueBERT     |  82%              | 89%               | 50%                |
