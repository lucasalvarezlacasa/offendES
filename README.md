# OffendES

_⚠️ This work constitutes the final practice of the subject "Redes Neuronales para PLN" of my Másters
at [UNED](https://www.uned.es/universidad/inicio/en/estudios/masteres/master-universitario-en-tecnologias-del-lenguaje.html?idContenido=1).
This is why some of the comments in the notebooks will be in Spanish.
Due to this, copying any of the work presented here and use it for the same subject is completely prohibited._

## Introduction

The proliferation of social media and the increase in citizen participation on these platforms in recent years have also
led to a rise in the exposure that many individuals face to various types of attacks based on gender, race, religion,
ideology, or other personal characteristics. For this reason, there has been growing interest in developing automated
systems capable of analyzing texts written on social media and detecting messages that can be classified as "hate
speech."

To carry out this task, a dataset generated from
the [OffendES: A New Corpus in Spanish for Offensive Language Research](https://aclanthology.org/2021.ranlp-1.123/)
collection is proposed. This dataset contains a total
of 13,368 training instances and 3,342 test instances. Each of these instances represents a post published on a social
media platform and is labeled with one of the following four possible classes:

- **Offensive, the target is an individual (OFP)**
- **Offensive, the target is a group of people or a collective (OFG)**
- **Non-offensive, but containing inappropriate or insulting language (NOM)**
- **Non-offensive (NO)**

## Task

A thorough and in-depth analysis must be conducted on the task, the associated
dataset, the proposed architectures, and the obtained results. The following steps should be followed:

🔍 **Data Analysis and Text Representation:** Perform an initial exploratory analysis of the dataset, including
visualizations, text analysis, class balance assessment, and issue identification with proposed solutions. Data cleaning
and preprocessing should be carried out using normalization techniques suitable for machine learning. Various text
representation methods (one-hot encoding, weighting schemes, embeddings, etc.) should be explored, justifying the choice
of each technique.

📏 **Baseline Definition:** Establish reference results using simple algorithms to compare against more complex models.
Recommended baselines include classical machine learning techniques such as Naïve Bayes, Logistic Regression, Decision
Trees, Random Forests, and SVMs. The rationale for selecting these algorithms should be explained, along with the
obtained results.

🧠 **Deep Learning Models:** Train and validate neural network-based models, defining and justifying the proposed
architectures. All types of architectures studied in the course should be explored (dense, convolutional, recurrent,
contextual models, etc.). Alternative models and innovative approaches are encouraged, even if results are not always
positive. Specific aspects of the task (e.g., transfer learning for a related dataset) should also be considered.

📊 **Result Analysis:** Evaluate the learning process, hyperparameters, and performance of each model individually and
comparatively. A critical analysis should be conducted to explain the results, concluding with a summary that compares
all explored models.

## Project

The project is organized into the following folders and files:

```
└── challenge
    ├── data
    │   └── dataset
    │       ├── OffendES
    │       │   ├── test.csv
    │       │   └── train.csv
    │       └── OffendES_dataset.zip
    ├── experiment_output
    │   ├── 1_data_exploration
    │   │   ├── v1
    │   │   │   ├── test.csv
    │   │   │   ├── train.csv
    │   │   │   └── val.csv
    │   │   ├── v2
    │   │   │   ├── test.csv
    │   │   │   ├── train.csv
    │   │   │   └── val.csv
    │   │   ├── v3
    │   │   │   ├── test.csv
    │   │   │   ├── train.csv
    │   │   │   └── val.csv
    │   │   ├── v4
    │   │   │   ├── test.csv
    │   │   │   ├── train.csv
    │   │   │   └── val.csv
    │   │   ├── v5
    │   │   │   ├── test.csv
    │   │   │   ├── train.csv
    │   │   │   └── val.csv
    │   │   ├── v6
    │   │   │   ├── augments.json
    │   │   │   ├── test.csv
    │   │   │   ├── train.csv
    │   │   │   └── val.csv
    │   │   ├── v7
    │   │   │   ├── test.csv
    │   │   │   ├── train.csv
    │   │   │   └── val.csv
    │   │   └── v8
    │   │       ├── test.csv
    │   │       ├── train.csv
    │   │       └── val.csv
    │   ├── 2_ml_baselines
    │   ├── 3_light_dl_approaches
    │   └── 4_dl_approaches
    │       ├── aux_chatgpt_orig_test_set_subset.tsv
    │       ├── aux_chatgpt_v1_test_set_subset.tsv
    │       ├── orig_test_set_chatgpt_classif_results_binary.tsv
    │       ├── orig_test_set_chatgpt_classif_results.tsv
    │       ├── orig_test_set_gpt4_classif_results.tsv
    │       ├── test_set_v1_chatgpt_classif_results_binary.tsv
    │       ├── test_set_v1_chatgpt_classif_results.tsv
    │       ├── test_set_v1_gpt4_classif_results.tsv
    │       └── test_set_v4_gpt4_classif_results.tsv
    │   └── 5_results_and_conclusions
    ├── notebooks
    │   ├── 1_data_exploration.ipynb
    │   ├── 2_ml_baselines.ipynb
    │   ├── 3_light_dl_approaches.ipynb
    │   ├── 4_dl_approaches.ipynb
    │   └── 5_results_and_conclusions.ipynb
    ├── README.md
    └── resources
        └── zero-shot-classification-alternatives.png
```

### Overview

- **data**: Stores external information that was not generated by us, such as the original dataset.
- **experiment_output**: Stores outputs generated in the notebooks, allowing them to be reused in subsequent flows. For
  example, the first notebook generates datasets that are later consumed by other notebooks for different models.
- **notebooks**: The different parts of the activity are separated into notebooks for better organization and separation
  of code.
- **resources**: Stores reference materials, such as images.

### Implementation

To properly separate the logic and reuse outputs from some flows in others (such as dataset processing and cleaning), I
will partition the implementation into multiple Jupyter Notebooks (see the [Notebook Index](#notebook-index)).

Each notebook will focus on a specific task, such as data analysis and processing, defining simple baseline models,
implementing deep learning models based on RNNs or CNNs, and finally, more complex models like contextual models,
Transformer-based architectures, and Large Language Models (LLMs).

Additional considerations:

- 💡 To limit the scope of this project, I will base my approach on state-of-the-art architectures for this task.
- 👎 I will define multiple dataset versions based on the representation needs of each model or
  augmentation techniques, but I will use only one cleanup version per case.
- 👍 For computationally expensive models, I will store test predictions to allow for more detailed analysis if
  necessary. This applies to cases where we use LLMs, for example, in _zero-shot_ or _few-shot_ mode.

### Notebook Index

The table below presents the set of notebooks used in this work, along with their objectives.

| File Name                       | Objective                                            | Description                                                                                                                                                                                                                                   |
|---------------------------------|------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1_data_exploration.ipynb        | Data analysis, exploration, and preprocessing        | Loads the dataset, performs initial exploration and data preprocessing, analyzes distribution and class balance, and proposes cleaning mechanisms.                                                                                            |
| 2_ml_baselines.ipynb            | ML baselines: Naive Bayes, SVM, Random Forests, etc. | Defines baselines mainly through training and validating simple Machine Learning models.                                                                                                                                                      |
| 3_light_dl_approaches.ipynb     | Deep Learning Models: Dense, RNN, CNN                | Trains and validates neural network-based systems, starting with dense networks, CNNs, and RNNs.                                                                                                                                              |
| 4_dl_approaches.ipynb           | Deep Learning Models: Transformers, contextual, LLMs | Trains and validates systems based on contextual models, using Transformer-based architectures and some _zero-shot_ and _few-shot_ tests with LLMs.                                                                                           |
| 5_results_and_conclusions.ipynb | Comparative analysis of results                      | Compares the performance of models trained in previous flows, providing a comparison of metrics (_accuracy_, _precision_, _recall_, and _f1_). Based on the comparison, a recommendation is made on the best model for production deployment. |

### Running Notebooks

Some notebooks were executed on my local laptop, while others required training on Google Colab due to GPU acceleration
needs. However, they are set up to run in both environments, with paths configured by the `RouteResolver` class.

To run the notebooks locally, it is necessary to set up an environment using `conda` or `venv` and install all
dependencies imported in the various notebooks. 