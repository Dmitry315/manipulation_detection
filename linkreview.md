# Обзор статей
| # | Название | Ссылка | О чём | Модель | Задача |
| - | -------- | ------ | ----- | ------ | ------ |
| 1 | PeerDA: Data Augmentation via Modeling Peer Relation for Span Identification Tasks | [link](https://arxiv.org/pdf/2210.08855.pdf) | Выделяются фрагменты текста, классифицируются и выделяются связи между фрагментами текста | BERT, RoBERTa | span id + tagging |
| 2 | Findings of the Shared Task on Offensive Span Identification from Code-Mixed Tamil-English Comments | [link](https://arxiv.org/pdf/2205.06118.pdf) | Выделяются оскорбительные фрагменты в тексте для соревнования https://competitions.codalab.org/competitions/36395 | RoBERTa | span id |
| 3 | 3218IR at SemEval-2020 Task 11: Conv1D and Word Embedding in Propaganda Span Identification at News Articles | [link](https://aclanthology.org/2020.semeval-1.225.pdf) | Выделяются манипулятивные фрагменты с помощью свёрточных сетей для соревнования SemEval-2020 Task 11 | Conv1D | span id |
| 4 | Dissecting Span Identification Tasks with Performance Prediction | [link](https://arxiv.org/pdf/2010.02587.pdf) | Рассматривается задача span id, оценка качества моделей | BERT Feat LSTM CRF | span id, metrics |
| 5 | Automatic identification of cited text spans: a multiclassifier approach over imbalanced dataset | [link](https://www.researchgate.net/profile/Chengzhi-Zhang-2/publication/324817301_Automatic_identification_of_cited_text_spans_a_multi-classifier_approach_over_imbalanced_dataset/links/5ae8198a45851588dd7f991d/Automatic-identification-of-cited-text-spans-a-multi-classifier-approach-over-imbalanced-dataset.pdf) | Решают задачу span id для суммаризации статей | Decision Tree, Logreg, SVM(linear, RBF) | span id |
| 6 | Poli2Sum@CL-SciSumm-19: Identify, Classify, and Summarize Cited Text Spans by means of Ensembles of Supervised Models | [link](https://www.researchgate.net/profile/Moreno-La-Quatra/publication/335079246_Poli2SumCL-SciSumm-19_Identify_Classify_and_Summarize_Cited_Text_Spans_by_means_of_Ensembles_of_Supervised_Models/links/5d4d872d92851cd046afc453/Poli2SumCL-SciSumm-19-Identify-Classify-and-Summarize-Cited-Text-Spans-by-means-of-Ensembles-of-Supervised-Models.pdf) | Выделяются фрагменты цитируемого текста | MLP, GradBoost | span id |
| 7 | DOSA: Dravidian Code-Mixed Offensive Span Identification Dataset | [link](https://aclanthology.org/2021.dravidianlangtech-1.2.pdf) | Предлагается датасет для выделения оскорбительных фрагментов текста | multilingual BERT, DistilBert, XML-RoBERTa | span id dataset |
| 8 | A Cross-Task Analysis of Text Span Representations | [link](https://arxiv.org/pdf/2006.03866.pdf) | Рассматривают различные постановки задач для span id | BERT, RoBERTa, SpanBERT, XLNet | span id |
| 9 | SpanBERT: Improving Pre-training by Representing and Predicting Spans | [link](https://arxiv.org/pdf/1907.10529.pdf) | Предлагается модель на основе BERT для выделения фрагментов | SpanBERT | span id |
| 10 | WLV-RIT at SemEval-2021 Task 5: A Neural Transformer Framework for Detecting Toxic Spans | [link](https://arxiv.org/pdf/2104.04630.pdf) | Решается задача выделения оскорбительных фрагментов для соревнования SemEval-2021 Task 5 | BERT, RoBERTa | span id |
| 11 | Improving the Efficiency of Grammatical Error Correction with Erroneous Span Detection and Correction | [link](https://arxiv.org/pdf/2010.03260.pdf) | В статье используют выделение фрагментов для исправления ошибок в тексте | BERT | span id |
| 12 | Sent2Span: Span Detection for PICO Extraction in the Biomedical Text without Span Annotations | [link](https://arxiv.org/pdf/2109.02254.pdf) | Решают задачу выделения фрагментов для биомедицинских текстов без разметки |  | unsupervised span id |
| 13 | Detect and Classify – Joint Span Detection and Classification for Health Outcomes | [link](https://arxiv.org/pdf/2104.07789.pdf) | Выделяются фрагменты, которые указывают на последствия для здоровья | BERT | span id |
| 14 | Data Augmentation with Dual Training for Offensive Span Detection | [link](https://aclanthology.org/2022.naacl-main.185.pdf) | Выделяются фрагменты c помощью GPT | GPT-2 | span id |
| 15 | Automatic Detection of Sentence Fragments | [link](https://aclanthology.org/P15-2099.pdf) | Используются синтаксические деревья и части речи, чтобы выделять фрагменты | syntax tree | span id |
| 16 | SemEval-2021 Task 5: Toxic Spans Detection | [link](https://aclanthology.org/2021.semeval-1.6.pdf) | Выделяются оскорбительные фрагменты текста для соревнования SemEval-2021 Task 5 (обзор решений) | BERT + CRF, RoBERTa | span id |
| 17 | SemEval-2020 Task 11: Detection of Propaganda Techniques in News Articles | [link](https://aclanthology.org/2020.semeval-1.186.pdf) | Описание задачи и обзор решений для соревнования SemEval-2020 Task 11 - детекция пропоганды и манипуляций в новостях | transformers ensamble (used CRF) | span id + tagging |
| 18 | ApplicaAI at SemEval-2020 Task 11: On RoBERTa-CRF, Span CLS and Whether Self-Training Helps Them | [link](https://aclanthology.org/2020.semeval-1.187.pdf) | Представляется решение для выделения фрагментов и классфикации типа пропоганды для соревнования SemEval-2020 Task 11 | RoBERTa-CRF | span id + tagging |
| 19 | A Unified MRC Framework for Named Entity Recognition | [link](https://arxiv.org/pdf/1910.11476.pdf) | Предлагается постановка задачи для решения задачи NER с вложенными фрагментами | LSTM, BERT | nested span id |
| 20 | Fine-Grained Analysis of Propaganda in News Articles | [link](https://arxiv.org/pdf/1910.02517.pdf) | Детекция манипуляций через выделение фрагментов и их классификации | BERT | span id + tagging |
| 21 | Simple yet Powerful: An Overlooked Architecture for Nested Named Entity Recognition | [link](https://aclanthology.org/2022.coling-1.184.pdf) | Решается задача NER с вложенными фрагментами | LSTM-CRF | nested span id |
| 22 | Quantifying Controversy on Social Media | [link](https://arxiv.org/pdf/1507.05224.pdf) | Строится граф разговора по теме и выделяются стороны которые противоречат друг другу | rule-based graph | polarization |
| 23 | APE: Argument Pair Extraction from Peer Review and Rebuttal via Multi-task Learning | [link](https://aclanthology.org/2020.emnlp-main.569.pdf) | Выделяются пары фрагментов аргументации | BERT + LSTM + CRF | span id |
| 24 | We Can Detect Your Bias: Predicting the Political Ideology of News Articles | [link](https://aclanthology.org/2020.emnlp-main.404.pdf) | Представляют датасет для классификации новостей по политическим идеологиям. | BERT | manipulatiton detection |
| 25 | SemEval-2007 Task 14: Affective Text | [link](https://aclanthology.org/S07-1013.pdf) | Датасет для классификации эмоций в заголовках новостей | - | emotion detection |
| 26 | name | link | about | model | task |
| 27 | name | link | about | model | task |
| 28 | name | link | about | model | task |
| 29 | name | link | about | model | task |
| 30 | name | link | about | model | task |
| 31 | name | link | about | model | task |
| 32 | name | link | about | model | task |
| 33 | name | link | about | model | task |
| 34 | name | link | about | model | task |
| 35 | name | link | about | model | task |
| 36 | name | link | about | model | task |
| 37 | name | link | about | model | task |
| 38 | name | link | about | model | task |
| 39 | name | link | about | model | task |
| 40 | name | link | about | model | task |

