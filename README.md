# Disturbed Statement Detection

Original dataset can be found here:
https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health

A text classification project that distinguishes statements expressing mental
distress from ordinary ones, using classical NLP and a tuned Logistic
Regression model. Includes a desktop GUI for testing the trained classifier.

> **This is not a diagnostic or screening tool.** It is a machine learning
> exercise built on social media text. Its predictions and identifications
> carry no clinical meaning and should not be used to assess anyone's 
> mental health.

## About the project

The dataset contains 52,680 statements labelled as *Disturbed* or *Normal*.
The source data separates distress into finer categories such as anxiety and
depression; this project collapses those into a single binary label, trading
diagnostic detail for a cleaner classification task.

Classes are imbalanced roughly 2:1 toward Disturbed, so the training set is
oversampled to an even split before fitting.

## Approach

**Preprocessing** — statements are lowercased and stripped of URLs, HTML tags,
@handles, punctuation, and tokens containing digits, then stopwords are
removed and the remainder is stemmed with the Snowball stemmer.

One deliberate choice: `no`, `not`, `nor` and `never` are removed from the
stopword list. Standard stopword filtering discards them, which would erase
the difference between "I am not okay" and "I am okay" — fatal for this task.

**Features** — CountVectorizer bag-of-words.

**Models** — Logistic Regression and Multinomial Naive Bayes, each tuned with
GridSearchCV over 5 folds, scored on weighted F1.

## Results

| Model | Test Accuracy | Test Weighted F1 |
|---|---|---|
| Tuned Logistic Regression | 0.945 | 0.945 |
| Baseline Logistic Regression | 0.943 | 0.944 |
| Tuned MultinomialNB | 0.894 | 0.892 |
| Baseline MultinomialNB | 0.892 | 0.888 |

Logistic Regression was selected for the application. Looking past the headline
figure, the two classes are not equally well predicted: Disturbed reaches 0.97
precision and 0.95 recall, while Normal manages 0.89 and 0.94. The model
over-flags rather than under-flags, which is the safer direction for this
subject matter but does mean ordinary statements are sometimes misclassified.

## Files

- `Text Classification.ipynb` — exploration, preprocessing, training, tuning,
  and model comparison
- `app.py` — Tkinter GUI that loads the trained model and classifies typed input
- `logistic_regression_model.pkl` — the tuned model, saved as a full pipeline
  so it accepts raw text directly
- `label_encoder.pkl` — maps the model's numeric output back to class names
- `count_vectorizer.pkl` — the fitted vectorizer, used in the notebook

## Getting started

Download the dataset and place `dataset.csv` in the project folder. Then
install the dependencies:

```bash
pip install -r requirements.txt
```

### Running the application

The application is Python source rather than a packaged executable, and needs
Python 3.12.

**From a terminal,** in the project folder:

```bash
python app.py
```

**In Visual Studio Code:** open the project folder, install the Python
extension, then press `Ctrl + Shift + P`, search for `Python: Select
Interpreter`, and choose the environment with the libraries installed. Open a
terminal with `Terminal → New Terminal` and run the command above.

Type a statement into the box and press Analyze. NLTK stopwords download
automatically on first run.