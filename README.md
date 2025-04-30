# Cyberbullying Classification (BERT + LSTM)

Detects cyberbullying types in tweets using a hybrid deep learning model: BERT embeddings feeding into stacked LSTM layers, trained and evaluated end‑to‑end in a single notebook.

See the notebook: [CyberBullyingLSTM.ipynb](CyberBullyingLSTM.ipynb)

## Overview
- **Goal:** Classify tweets into cyberbullying categories using modern NLP.
- **Approach:** Tokenize with `bert-base-uncased`, extract contextual embeddings via `TFBertModel`, then model sequence dynamics with two LSTM layers and a softmax classifier.
- **Outputs:** Validation accuracy, `classification_report`, and confusion matrix.

## Dataset
- **Source:** Kaggle — Andrew Mvd, “Cyberbullying Classification”
- **Link:** https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification
- **Key columns:** `tweet_text` (input), `cyberbullying_type` (label)
- **Local file expected:** `cyberbullying_tweets.csv` placed in the project root.

## Model Architecture
- **Tokenizer:** `BertTokenizer` (`bert-base-uncased`), `MAX_LEN = 128`.
- **Encoder:** `TFBertModel` sequence outputs.
- **Head:** `LSTM(64, return_sequences=True)` → `LSTM(32)` → `Dropout(0.3)` → `Dense(num_classes, activation='softmax')`.
- **Loss/Opt:** `sparse_categorical_crossentropy`, `Adam(lr=2e-5)`.
- **Train split:** `train_test_split(..., test_size=0.2, random_state=42)`.
- **Training:** `epochs=3`, `batch_size=16`.

## Preprocessing
- **Text cleaning:** Lowercase, remove URLs, non‑alphabetic chars, collapse spaces.
- **Label encoding:** `LabelEncoder` on `cyberbullying_type` → `label_enc`.

## Windows Setup
Use a virtual environment and install dependencies.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install tensorflow transformers scikit-learn pandas numpy matplotlib kaggle
```

## Download the Dataset (Kaggle)
Option 1 — Kaggle CLI (recommended for local Windows):

```powershell
# Put kaggle.json in %USERPROFILE%\.kaggle\
mkdir %USERPROFILE%\.kaggle 2> $null
copy "path\to\kaggle.json" "%USERPROFILE%\.kaggle\kaggle.json"

# Download to current folder
kaggle datasets download -d andrewmvd/cyberbullying-classification -p .

# Unzip (PowerShell)
powershell -Command "Expand-Archive -Path .\cyberbullying-classification.zip -DestinationPath ."
```

Option 2 — Google Colab (already included in the notebook): upload `kaggle.json`, run the Kaggle cells, then download `cyberbullying_tweets.csv` and place it locally.

## Run the Notebook
- Open [CyberBullyingLSTM.ipynb](CyberBullyingLSTM.ipynb) in VS Code or Jupyter.
- Ensure `cyberbullying_tweets.csv` is present in the workspace root.
- Run cells top‑to‑bottom. Training will produce metrics and plots.

## Results
- **Validation accuracy:** Printed after evaluation.
- **Detailed metrics:** `classification_report(y_val, y_pred)`.
- **Visualization:** Confusion matrix plotted with `matplotlib`.

## Notes
- The first cells include Colab‑specific Kaggle setup. On Windows, prefer the Kaggle CLI instructions above.
- GPU acceleration (optional) may require installing CUDA/CuDNN compatible with your TensorFlow version.

## Citation
If you use this dataset/model, please cite:

Andrew Mvd. Cyberbullying Classification. Kaggle. https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification

## Acknowledgements
- Hugging Face `transformers` and TensorFlow Keras.
- Scikit‑learn for metrics and label encoding.