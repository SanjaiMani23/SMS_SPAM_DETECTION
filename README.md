# SMS Spam Detection with a GRU (TensorFlow / Keras)

A deep learning pipeline that classifies SMS messages as **spam** or **ham** (legitimate) using a Gated Recurrent Unit (GRU) network. The project covers the full workflow: data loading and cleaning, text preprocessing, tokenization and padding, model selection, training, evaluation, and predictions on unseen messages.

## Problem Statement

A mobile communication company wants to identify spam messages automatically using deep learning. This project uses a manageable subset of approximately 2,000 labelled messages from the SMS Spam Detection dataset.

## Dataset

- **Source:** [SMS Spam Detection Dataset on Kaggle](https://www.kaggle.com/datasets/vishakhdapat/sms-spam-detection-dataset)
- **File:** `spam_sms.csv` (5,572 messages: 747 spam, 4,825 ham)
- **Columns:** `v1` (label: `ham` or `spam`), `v2` (message text)
- **Subset used:** 2,000 messages, stratified so the spam ratio is preserved (about 12.4% spam)

## Project Tasks

1. **Load and preprocess the data:** read the CSV, check for missing values, drop empty and duplicate messages, take a stratified 2,000-message subset, and clean the text (lowercasing, replacing URLs, e-mails, prices and numbers with placeholder tokens, and removing punctuation). Stop words are kept, since words such as "you", "now" and "free" are useful spam signals.
2. **Tokenize and pad:** split the data 70/10/20 into train, validation and test sets (stratified). A Keras `Tokenizer` with a 3,000-word vocabulary is fitted on the training set only. Sequences are padded to the 95th-percentile message length.
3. **Model:** a GRU-based classifier with an embedding layer, a GRU layer with 64 units, dropout, a dense layer and a sigmoid output. The GRU is compared against a Simple RNN and an LSTM on the validation set.
4. **Train and evaluate:** training uses early stopping on validation loss and class weights to handle the spam minority. Accuracy, precision, recall and F1-score are reported on the held-out test set.
5. **Confusion matrix and predictions:** the confusion matrix and misclassified messages are displayed, and the model predicts spam probability for unseen SMS messages.

## Why GRU?

| Model | Consideration |
|---|---|
| Simple RNN | Suffers from vanishing gradients, so it struggles to retain early words in a message. |
| LSTM | Handles long-range dependencies well, but has more parameters, which makes it slower and more prone to overfitting on small data. |
| **GRU** | Keeps gating to address vanishing gradients, with fewer parameters than an LSTM. It suits short SMS texts, where spam cues are local phrases such as "call now" or "claim your prize". |

## Model Architecture

```
Embedding (vocab 3000, dim 64, mask_zero=True)
GRU (64 units)
Dropout (0.4)
Dense (32, ReLU)
Dropout (0.3)
Dense (1, sigmoid)   -> P(spam)
```

- **Loss:** binary cross-entropy
- **Optimizer:** Adam (learning rate 0.001)
- **Regularisation:** dropout, early stopping (patience 4), class weights

## Results

Test set (400 messages, spam is the positive class):

| Metric | Score |
|---|---|
| Accuracy | 98.5% |
| Precision | 95.8% |
| Recall | 92.0% |
| F1-score | 93.9% |

Confusion matrix:

|  | Predicted ham | Predicted spam |
|---|---|---|
| **Actual ham** | 348 | 2 |
| **Actual spam** | 4 | 46 |

Results use a fixed random seed (42). Small differences may appear on other environments or TensorFlow versions, because the test set contains only 50 spam messages.

## Limitations

- The model learns the spam styles present in the training data. The dataset is dominated by older prize and "text to enter" messages, so differently worded spam, such as a bank-phishing alert without typical trigger words, can be missed.
- Only a 2,000-message subset is used, so the spam examples are limited.
- The model should be retrained periodically on recent messages, because spam tactics change over time.

## Project Structure

```
.
├── README.md
├── spam_sms.csv                  # dataset (download from Kaggle)
├── SMS_Spam_Detection_GRU.ipynb  # notebook with all five tasks
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

`requirements.txt`:

```
tensorflow>=2.15
pandas
numpy
scikit-learn
matplotlib
```

## How to Run

### Option 1: Google Colab (recommended)

1. Open [Google Colab](https://colab.research.google.com) and create a new notebook, or upload `SMS_Spam_Detection_GRU.ipynb`.
2. Upload `spam_sms.csv` using the folder icon in the left sidebar.
3. Select **Runtime → Run all**. A GPU is not required.

### Option 2: Local machine

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
jupyter notebook SMS_Spam_Detection_GRU.ipynb
```

Place `spam_sms.csv` in the same folder as the notebook before running.

## Predicting on New Messages

After training, pass any list of strings to the `predict_sms` function:

```python
predict_sms([
    "WINNER!! You have been selected to receive a £900 prize. Call now to claim.",
    "Are we still meeting for lunch tomorrow?"
])
```

Each message returns a spam probability and a label (`SPAM` or `HAM`). The default threshold is 0.5 and can be changed with the `threshold` argument.

## Reproducibility

- Random seed: 42 (set for NumPy, TensorFlow and scikit-learn splits)
- Stratified splits preserve the spam ratio in every subset

## Acknowledgements

- Dataset: SMS Spam Collection, published on Kaggle by vishakhdapat. It derives from the UCI SMS Spam Collection.

## License

Add a license here (for example MIT) before publishing.
