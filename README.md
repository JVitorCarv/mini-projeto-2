# 💬 ChatGPDiss - ChatGPT Tweet Sentiment Analysis 🧠

This project trains and evaluates deep learning models for **multiclass tweet sentiment analysis** (Bad, Neutral and Good), using **PyTorch**, **Streamlit** for the interative interface, and visualizations with `matplotlib` and `great_tables`.

---

## 📁 Project Structure

```
.
├── graphs/                    # Saved plots
├── models/                    # Trained models (.pt)
├── neural_networks/           # Model architecture definitions (RNN, LSTM)
├── nlp/                       # NLP preprocessing utilities
├── presentation/              # Streamlit app interface
│   └── app.py
├── results/                   # Output from experiments
├── notebook.ipynb             # Main notebook with training, evaluation, and visualization
├── pyproject.toml             # Project metadata and dependencies
└── uv.lock                    # uv dependency lock file
```

---

## 🛠️ Requirements

Install dependencies using [uv](https://github.com/astral-sh/uv), a fast Python package manager:

```bash
uv venv                     # Create and activate virtual environment
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
uv sync                     # Install dependencies
```

> To save tables with `great_tables`, the `weasyprint` package is also required (and may need system dependencies like Cairo and Pango).

All of the data is imported using `kagglehub`.

---

## 🚀 How to Run

1. Make sure you have installed all dependencies using `uv` as described above.
2. Run the Streamlit app:

```bash
streamlit run presentation/app.py
```

3. A browser window should open at `http://localhost:8501` showing the interactive interface.

---

## 🔄 How to Reproduce Results

- **Training**: run `notebook.ipynb` to preprocess the data, train both RNN and LSTM classifiers for sentiment analysis, and save the best model checkpoints to the `models/` folder.
- **Evaluation**: accuracy is computed on the training and test sets at each epoch, and key metrics are visualized using `matplotlib`.
- **Classification Report**: rendered as styled tables using `great_tables`, summarizing precision, recall, and F1-score for each sentiment class.

---

## 📊 Results

- **Final accuracy on holdout**: 91.25%
- **Best model**: `LSTM`
- **F1-Scores**:
  - Bad: 0.93
  - Neutral: 0.88
  - Good: 0.92
