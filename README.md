AGNewsNLP: Multi‑Class News Topic Classification
This project builds a full NLP pipeline to classify news articles from the AG News dataset into four topics: World, Sports, Business, and Sci/Tech.
It includes data exploration, preprocessing, feature extraction with TF‑IDF and Word2Vec, multiple machine‑learning models, evaluation and benchmarking, unit‑test style checks, and a Streamlit web application.

1. Project structure
text
ag_news_nlp_project/
│
├─ data/
│   └─ train.csv
│
├─ notebooks/
│   └─ 01_ag_news_pipeline.ipynb
│
├─ src/
│   ├─ data_loader.py
│   ├─ preprocess.py
│   ├─ features.py
│   ├─ models.py
│   ├─ train.py
│   ├─ evaluate.py
│   ├─ visualization.py
│   └─ app_streamlit.py
│
├─ src/tests/
│   ├─ test_preprocess.py
│   ├─ test_features.py
│   └─ test_models.py
│
├─ reports/
│   ├─ figures/
│   └─ final_report.pdf
│
├─ tfidf_vectorizer.pkl
├─ best_model.pkl
├─ requirements.txt
└─ README.md
data/: AG News CSVs.

notebooks/: Colab/Jupyter notebooks used for EDA and experiments.

src/: Reusable Python modules for the pipeline and the Streamlit app.

src/tests/: Simple unit‑test style checks for preprocessing, features, and models.

reports/: Final written report and generated figures (EDA plots, PCA, confusion matrices).

2. Dataset
The project uses the AG News dataset, a standard text‑classification benchmark with 4 news categories.
Each record contains a news title, description, and a label indicating the topic.

In this repository:

train.csv contains the training articles and labels.

The notebook creates internal train/validation/test splits (80/10/10) for model development.

3. Methods
3.1 Preprocessing
The preprocessing pipeline includes:

Lowercasing text

Removing URLs, numbers, and punctuation

Tokenization using NLTK

Stopword removal

(Optional) Lemmatization

Cleaned text is stored in a clean_text column and tokenized text in a tokens column.

3.2 Feature extraction
Two main feature types are used:

TF‑IDF (TfidfVectorizer)

Unigrams and bigrams

Limited vocabulary size for efficiency

Word2Vec

Trained on the tokenized AG News corpus

Each document represented by the mean of its word vectors

3.3 Dimensionality reduction (PCA)
Principal Component Analysis (PCA) is applied to both TF‑IDF and Word2Vec document vectors to produce 2‑D visualizations that show how well the four classes separate in feature space.

4. Models
Three model families are implemented:

Multinomial Naive Bayes (TF‑IDF features)

Logistic Regression (TF‑IDF features)

Feed‑forward Neural Network (Word2Vec document vectors)

The neural network is a small fully connected network with ReLU activations, dropout regularization, and a softmax output over the four classes.

5. Training and evaluation
The training and evaluation scripts:

Split the data into train/validation/test sets (stratified).

Train each of the three models.

Evaluate on validation and test sets with:

Accuracy

Macro and weighted F1‑score

Per‑class precision, recall, and F1

Results are summarized in tables and confusion‑matrix heatmaps.
The best‑performing model (typically Logistic Regression with TF‑IDF) is exported as best_model.pkl along with its tfidf_vectorizer.pkl.

6. Streamlit application
The Streamlit app (app_streamlit.py) provides an interactive interface:

Input: a news headline or short article text.

Output:

Predicted topic (World, Sports, Business, Sci/Tech)

Class probabilities for all four labels

The app loads the saved TF‑IDF vectorizer and best model and uses the same preprocessing logic as in training.

Running the app locally
Install dependencies:

bash
pip install -r requirements.txt
(or at minimum: pip install streamlit scikit-learn pandas numpy and any others you need.)

Make sure app_streamlit.py, tfidf_vectorizer.pkl, and best_model.pkl are in the same folder.

Run:

bash
streamlit run app_streamlit.py
Open the URL shown in the terminal (usually http://localhost:8501) in your browser.

7. Unit tests
Basic unit tests are included under src/tests/:

test_preprocess.py

Verifies that cleaning removes URLs, numbers, and punctuation.

Checks that tokenization returns non‑empty lists for typical inputs.

test_features.py

Ensures TF‑IDF vectors are not empty.

Confirms Word2Vec document vectors have the expected dimensions.

test_models.py

Trains each model on a small subset.

Checks that fit and predict run without errors and predictions have correct shapes.

These tests demonstrate correctness of the main pipeline components and align with the “UnitTest” requirement.

8. How to reproduce the project (short guide)
Open the notebook in the notebooks/ folder and run cells from top to bottom:

Load data, perform EDA and visualization.

Run preprocessing and feature extraction.

Train and evaluate all models.

Save the best model and vectorizer (best_model.pkl, tfidf_vectorizer.pkl).

Run tests (optional but recommended):

bash
pytest src/tests
Launch the Streamlit app as described above.

Write or review the report in reports/final_report.pdf summarizing:

Dataset and task

Methods and models

Evaluation results and comparisons

Limitations and future work

9. Academic notes
This project is designed for academic submission and explicitly addresses:

Data: clear description of AG News and its labels.

Preprocess: text cleaning, tokenization, feature engineering (TF‑IDF, Word2Vec), and data splitting.

Model: at least three distinct models.

Evaluate: complex benchmarking with multiple metrics and confusion matrices.

Application: packaged into an interactive Streamlit app, with correct use of technical terminology.
