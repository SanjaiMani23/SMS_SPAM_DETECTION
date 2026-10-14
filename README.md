# SMS Spam Detection with a GRU (TensorFlow / Keras)

A deep learning pipeline that classifies SMS messages as **spam** or **ham** (legitimate) using a Gated Recurrent Unit (GRU) network. The project follows a full workflow: data loading and cleaning, text preprocessing, tokenization and padding, model selection, training, evaluation, and predictions on unseen messages.

## Problem Statement

A mobile communication company wants to identify spam messages automatically using deep learning. This project uses a manageable subset of approximately 2,000 labelled messages from the SMS Spam Detection dataset.

## Dataset

- **Source:** [SMS Spam Detection Dataset on Kaggle](https://www.kaggle.com/datasets/vishakhdapat/sms-spam-detection-dataset)
- **File:** `spam_sms.csv` (5,572 messages, 747 spam and 4,825 ham)
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
