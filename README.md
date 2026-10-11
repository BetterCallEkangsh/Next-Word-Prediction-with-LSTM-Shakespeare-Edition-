# 🎭 Next Word Prediction with LSTM (Shakespeare Edition)

> *"To be or not to ___"*

A next-word prediction project trained on Shakespeare's **Hamlet**. It tokenizes the play, builds n-gram training sequences, trains a stacked **LSTM** (with a **GRU** variant for comparison) in TensorFlow/Keras, and serves predictions through a **Streamlit** web app.

---

## ✨ Features

- 📚 Trained on *Hamlet* from the NLTK Gutenberg corpus
- 🧠 Stacked LSTM with Dropout, plus a GRU variant for comparison
- 🔡 Word-level tokenization with a 4,818-token vocabulary
- 🖥️ Streamlit app for interactive predictions
- 💾 Pre-trained model and tokenizer included

---

## 🗂️ Project Structure

```
├── app.py                    # Streamlit web app
├── notebooks/
│   └── experiments.ipynb     # Data prep, training, GRU comparison
├── hamlet.txt                # Training corpus (Hamlet, Project Gutenberg)
├── LSTM_Word_Predictor.h5    # Trained LSTM model (used by the app)
├── GRU Predictor.h5          # Trained GRU model (comparison)
├── tokenizer.pkl             # Fitted Keras tokenizer
├── requirements.txt          # Dependencies
└── README.md
```

> Adjust the paths above to match your actual repo layout. `app.py` expects `LSTM_Word_Predictor.h5` and `tokenizer.pkl` in the folder it is run from.

---

## 🏗️ How It Works

1. **Data collection:** Hamlet is loaded from NLTK's Gutenberg corpus and saved to `hamlet.txt`.
2. **Tokenization:** Text is lowercased and fed to a Keras `Tokenizer` (vocabulary size: **4,818**, including the padding index).
3. **N-gram sequences:** For every line, progressively longer prefixes are created, e.g. `[w1, w2]`, `[w1, w2, w3]`, and so on. This yields **25,732** sequences.
4. **Padding:** Sequences are pre-padded to a length of **14**. The last token is the label and the first 13 are the input.
5. **Labels:** One-hot encoded over the vocabulary.
6. **Split:** 80/20 train/validation split (`random_state=42`).
7. **Inference:** The input text is tokenized, truncated or padded to 13 tokens, and the `argmax` of the softmax output is mapped back to a word.

### Model Architecture

| Layer | Details |
|-------|---------|
| Embedding | vocab size 4,818, 100 dimensions |
| LSTM | 150 units, `return_sequences=True` |
| Dropout | 0.2 |
| LSTM | 100 units |
| Dense | 4,818 units, softmax |

The GRU variant swaps both LSTM layers for GRU layers of the same size.

- **Loss:** categorical cross-entropy
- **Optimizer:** Adam
- **Metric:** accuracy

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/BetterCallEkangsh/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Create an environment

Using conda:

```bash
conda create -n nextword python=3.11 -y
conda activate nextword
```

Or with venv:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> The saved `.h5` models were created with **Keras 3 (TensorFlow 2.16+)**. Use a similar TensorFlow version to avoid load errors. Run `pip show tensorflow` to see which one you have, and pin it in `requirements.txt` for reproducibility.

### 4. Run the app

```bash
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`), type a phrase, and click **Predict Next Word**.

---

## 🔁 Retraining

Open `experiments.ipynb` and run the cells in order. At the end it saves:

- `LSTM_Word_Predictor.h5` (LSTM model)
- `tokenizer.pkl` (fitted tokenizer)
- `GRU Predictor.h5` (GRU model)

---

## 📊 Results & Known Limitations

Being upfront about how it actually performs:

| Model | Epochs | Train accuracy | Validation accuracy |
|-------|--------|----------------|---------------------|
| LSTM | 200 | ~82.6% | ~4.2% |
| GRU | 10 | ~80.6% | ~5.0% |

- **Heavy overfitting.** Training accuracy is high while validation accuracy stays around 4 to 5%, and validation loss climbs instead of falling. The models largely memorize a single 25k-sequence corpus rather than learning language.
- **Early stopping is not applied yet.** An `EarlyStopping` callback (`val_loss`, `restore_best_weights=True`) is defined in the notebook but not yet passed to `model.fit(...)`. Adding `callbacks=[earlystopper]` is the first item on the to-do list.
- **Tiny, archaic corpus.** Hamlet is one play with Early Modern English spelling (`haue`, `vpon`, `selfe`), so predictions can look odd on modern text.
- **Greedy prediction.** The app returns only the single top word, with no top-k and no sampling.
- **Line-level sequences.** Sequences are built per line, so context never carries across line breaks.
- **Out-of-vocabulary words are dropped** by the tokenizer.

---

## 🔮 Future Improvements

- [ ] Pass `EarlyStopping` to `fit()` and use fewer epochs
- [ ] Add more regularization (higher dropout, smaller model)
- [ ] Train on more Shakespeare works or a larger corpus
- [ ] Show top-k predictions with probabilities in the UI
- [ ] Temperature sampling for more varied output
- [ ] Save models in the native `.keras` format instead of legacy `.h5`
- [ ] Deploy on Streamlit Community Cloud or Hugging Face Spaces

---

## 🛠️ Tech Stack

`Python` · `TensorFlow / Keras` · `NumPy` · `Pandas` · `scikit-learn` · `NLTK` · `Streamlit`

---

## 👤 Author

**Ekangsh**
GitHub: [@BetterCallEkangsh](https://github.com/BetterCallEkangsh)

---

## 📄 License

*Hamlet's text is in the public domain via Project Gutenberg.*
