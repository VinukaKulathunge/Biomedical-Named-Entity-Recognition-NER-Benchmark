# BC5CDR Disease Named Entity Recognition (NER) Benchmark

This project is a complete, production-ready, modular PyTorch named entity recognition (NER) suite designed to recognize **disease entities** from the **BC5CDR** biomedical corpus. It compares **30 pipeline configurations** (3 Tokenizers × 5 Embeddings × 2 Classification Layers) using a Bidirectional LSTM (BiLSTM) sequence model.

---

## 1. Project Architecture

The codebase is highly modular, well-documented, and adheres to modern NLP software engineering principles:

```
ner_disease_recognition/
├── data/                     # BC5CDR dataset files (auto-downloaded/synthesized)
├── tokenizer_custom/         # Word and subword segmentation
│   ├── __init__.py
│   ├── whitespace.py         # Whitespace-based splitting
│   ├── nltk_tokenizer.py     # NLTK Treebank splitter with character span offsets
│   └── bpe_tokenizer.py      # HuggingFace BertTokenizerFast WordPiece subwords
├── embeddings/               # Dense representational vectors
│   ├── __init__.py
│   ├── word2vec_embed.py     # Gensim Word2Vec trained on corpus (100% coverage)
│   ├── glove_embed.py        # 50d GloVe parsing with auto corpus Word2Vec fallback
│   ├── fasttext_embed.py     # Local subword FastText for rare clinical terms
│   ├── elmo_embed.py         # Custom Contextual BiLSTM Language representation in PyTorch
│   └── bert_embed.py         # Frozen feature-extractor BERT with character-span alignment
├── models/                   # Sequence tagging architectures
│   ├── __init__.py
│   ├── bilstm_crf.py         # BiLSTM + pytorch-crf Viterbi decoder
│   └── bilstm_softmax.py     # BiLSTM + Softmax Cross-Entropy decoder
├── pipeline/                 # Pipeline orchestration
│   ├── __init__.py
│   └── ner_pipeline.py       # Manages tensor padding, collation, training, & metrics
├── evaluation/               # Validation metrics
│   ├── __init__.py
│   └── metrics.py            # Micro Precision, Recall, and F1 (ignores 'O' class)
├── experiments/              # Benchmark orchestrator
│   ├── __init__.py
│   └── run_all.py            # Runs grid benchmark, saves CSV, & writes Markdown report
├── main.py                   # Main CLI entry point
├── requirements.txt          # Python package requirements
└── README.md                 # Project documentation (this file)
```

---

## 2. Key Technical Innovations

### 2.1 Character-Based Token Alignment
Different tokenizers segment medical terms differently than the original space-separated CoNLL TSV files. This changes the indices of target labels. 
To resolve this:
1. We reconstruct the raw sentence string while recording a precise mapping of each character index back to its original CoNLL token index.
2. We run the selected tokenizer and capture the start and end character indices (spans) for the newly split tokens.
3. We align the BIO tags to the new tokens: the first sub-token overlapping an original word gets its raw tag (e.g. `B-Disease`), and any subsequent sub-tokens get `I-Disease`.
4. Special tokens (`[CLS]`, `[SEP]`) are mapped to `O` and masked out of evaluations.

### 2.2 Pre-trained Features & Out-of-the-Box Fallbacks
To ensure the pipeline is 100% runnable without heavy manual file downloads:
* **Word2Vec & FastText:** Automatically train standard representations on the input corpus itself using `gensim` in seconds, guaranteeing 100% vocabulary coverage.
* **GloVe:** Automatically looks for a local copy of Stanford's `glove.6B.50d.txt` inside `data/` or root. If not found, it prints a warning and automatically falls back to training a 50-dimensional Word2Vec model on the corpus so the experiments run immediately.
* **ELMo Contextualized Representation:** Rather than using the deprecated `allennlp` library, we implement a beautiful multi-layer Bidirectional LSTM Language Model encoder in PyTorch that contextualizes word embeddings end-to-end.
* **BERT Contextual Feature-Extractor:** Uses a frozen pre-trained Hugging Face BERT model (`bert-base-uncased`) and extracts hidden states from the last layer. A dynamic character-offset mapping overlaps BERT subwords back to Whitespace/NLTK word tokens (averaging subwords when split), making BERT fully compatible with all three tokenizers.
* **Automatic Dataset Setup:** If the local dataset is missing from `data/`, the data loader automatically downloads the disease-split CoNLL files from Cambridge LTL's GitHub repository. If offline, it generates a synthetically realistic biomedical corpus containing real disease terms (e.g. *myocardial infarction, diabetes mellitus, renal injury*), ensuring zero crashes.

---

## 3. Installation & Setup

1. **Clone or Navigate to the Directory:**
   ```bash
   cd ner_disease_recognition
   ```

2. **Set up a Virtual Environment (Recommended):**
   ```bash
   python -m venv venv
   venv\Scripts\activate      # On Windows
   source venv/bin/activate    # On macOS/Linux
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 4. Usage Instructions

The project provides an interactive CLI through `main.py`.

### 4.1 Running the 30-Pipeline Benchmark Suite (Recommended)
This runs all 15 tokenizer-embedding combinations under both CRF and Softmax classifiers, prints a comparative console table, logs results to `results/f1_results.csv`, and writes a polished summary report to `results/report.md`.

* **Fast Benchmark Mode (Verifies compiling and logic on CPU in ~90s):**
  Uses a representative subset of the corpus and 2 epochs per model:
  ```bash
  python main.py --run-all --fast
  ```
  
* **Full Benchmark Mode (Trains on the full BC5CDR dataset for academic results):**
  Uses the complete dataset and 5 epochs per model:
  ```bash
  python main.py --run-all --full
  ```

### 4.2 Running a Single Custom Configuration
You can configure and train a single pipeline using custom parameters:
```bash
python main.py --tokenizer NLTK --embedding Word2Vec --classifier CRF --epochs 3 --batch_size 16 --lr 0.001 --full
```

#### CLI Parameters:
* `--run-all`: Toggle grid search evaluation.
* `--tokenizer`: Choose from `Whitespace`, `NLTK`, `BPE`. (Default: `BPE`).
* `--embedding`: Choose from `Word2Vec`, `GloVe`, `FastText`, `ELMo`, `BERT`. (Default: `BERT`).
* `--classifier`: Choose from `CRF`, `Softmax`. (Default: `CRF`).
* `--epochs`: Number of epochs to train. (Default: `3`).
* `--batch_size`: Size of mini-batches. (Default: `16`).
* `--hidden_dim`: Hidden layer size of the BiLSTM. (Default: `64`).
* `--fast` / `--full`: Toggles fast representative subset training vs full dataset training. (Default: `--fast`).

---

## 5. Evaluation Details

* **Metric:** Micro F1-score computed over disease class tokens (`B-Disease` and `I-Disease`), strictly **ignoring the "O" (Outside) class** as requested.
* **Output Artifacts:**
  * `results/f1_results.csv`: Table of F1, Precision, Recall, and Runtime for all configurations.
  * `results/report.md`: A styled, comprehensive markdown analysis highlighting the best configuration, average performances by tokenizer, embedding, and classifier, and qualitative NLP takeaways.
