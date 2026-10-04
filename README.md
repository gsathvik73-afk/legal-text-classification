# Classification of Legal Text

Machine Learning mini-project, Team 9. The project classifies U.S. Supreme Court opinions into 13 legal issue areas using TF-IDF, LDA and Doc2Vec, each with Logistic Regression.

## Setup

Use Python 3.11 or 3.12. Install the packages once:

```bash
python -m pip install numpy==1.26.4 scipy==1.13.1 scikit-learn==1.5.2 gensim==4.3.3
```

## Run

First, download and prepare the data:

```bash
python prepare_data.py
```

Then run whichever model you want:

```bash
python tfidf_model.py
python lda_model.py
python doc2vec_model.py
```

The scripts use the same training, validation and test split. Each model prints its scores, saves results in `data/` and lets you type text for a prediction after training. Press Enter to finish. Doc2Vec can take tens of minutes on a CPU.

The `data/` folder is created locally and should stay out of Git.

Dataset: [textacy Supreme Court corpus](https://github.com/bdewilde/textacy-data/releases/tag/supreme_court_py3_v1.0). Reference study: Krithika Iyer, *Classification of Legal Text*.
