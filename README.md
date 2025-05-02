# Fake News Detection

A machine learning project to detect fake news articles using Logistic Regression and Random Forest classifiers. The project uses TF-IDF vectorization to transform news article content into numerical features.

##  Dataset

The dataset consists of two CSV files:

- `true.csv`: Real news articles
- `fake.csv`: Fake news articles

Each file includes:
- `title`: The headline of the news article
- `text`: The body of the news article

These datasets are merged and labeled (`1` for real, `0` for fake) before training.

## Technologies Used

- Python
- Pandas
- NumPy
- scikit-learn

##  Preprocessing Steps

1. Merge `fake.csv` and `true.csv` with appropriate labels.
2. Combine `title` and `text` into a single `content` field.
3. Convert text to lowercase and remove non-alphabetic characters.
4. Remove basic English stopwords.
5. Apply TF-IDF vectorization (top 5000 features).

##  Models Used

- **Logistic Regression** (`sklearn.linear_model.LogisticRegression`)
- **Random Forest Classifier** (`sklearn.ensemble.RandomForestClassifier`)

Both models are trained and evaluated on an 80/20 train-test split.

##  Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

##  How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fake-news-detector.git
   cd fake-news-detector
