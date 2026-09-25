# A Comparative Analysis of NLP and Machine Learning Approaches for Phishing Email Detection

Logistic Regression, Random Forest, and a fine-tuned BERT model trained and evaluated on the same
39,154-email corpus under identical preprocessing and metrics, to test whether transformer-based
deep learning is worth its cost over traditional ML for phishing detection.

**Jai Barber** — Department of Computer Science, Southern University and A&M College, Baton Rouge, LA

📄 **[Read the paper](paper/Jai_Barber_IEEE_ResearchPaper.pdf)** (IEEE format)

---

## Result

All three models reached **99% accuracy**. The interesting part is where they disagree.

| Model | Precision | Recall | F1 | Total errors | Training time |
|---|---|---|---|---|---|
| Logistic Regression | 0.99 | **1.00** | 0.99 | 63 | ~5 sec |
| Random Forest | 0.99 | 0.99 | 0.99 | 55 | ~2 min |
| BERT | **1.00** | 0.99 | 0.99 | **50** | ~30 min (T4 GPU) |

BERT won by 13 errors out of 7,831 test emails — and took **360× longer to train** to get there.

For a resource-constrained team, Logistic Regression is the practical choice. BERT earns its cost
only where false positives are expensive, or where phishing language closely mimics legitimate
business communication — which this corpus's spam-style phishing does not.

### Error tradeoff

| Model | TN | FP | FN | TP |
|---|---|---|---|---|
| Logistic Regression | 3,448 | 42 | **21** | 4,320 |
| Random Forest | 3,459 | 31 | 24 | 4,317 |
| BERT | 3,479 | **11** | 39 | 4,302 |

The two extremes optimize for opposite failures. **BERT minimizes false alarms** (11 FP) but misses
the most real phishing (39 FN). **Logistic Regression misses the least phishing** (21 FN) at the cost
of flagging more legitimate mail (42 FP).

For phishing detection a false negative is the costlier error — a missed attack reaches the user,
while a false positive costs someone a trip to the spam folder. On that criterion Logistic
Regression, the cheapest model, is also the best performer.

---

## Pipeline

The project runs in two notebooks, in order:

### Part 1 — [`AI_Phishing_Detection.ipynb`](AI_Phishing_Detection.ipynb)

Exploratory analysis and feature engineering. Establishes what actually separates the two classes,
then writes `phishing_cleaned.csv` to Google Drive.

### Part 2 — [`Phishing detection Part 2.ipynb`](Phishing%20detection%20Part%202.ipynb)

Reads that cleaned CSV, then trains and evaluates all three models on an identical 80/20 split.

```
CEAS_08.csv
     │
     ▼
┌─────────────────────────────┐
│ Part 1 — EDA                │
│  • class balance            │
│  • email length             │
│  • urgency keywords         │
│  • URL counts               │
│  • word frequency           │
└──────────────┬──────────────┘
               │  phishing_cleaned.csv  (via Google Drive)
               ▼
┌─────────────────────────────┐
│ Part 2 — Modeling           │
│  • 80/20 split (31,323 /    │
│    7,831), random_state=42  │
│  • TF-IDF: 5,000 features,  │
│    English stopwords        │
│  • Logistic Regression      │
│  • Random Forest (100 trees)│
│  • BERT (bert-base-uncased) │
│  • comparison charts        │
└─────────────────────────────┘
```

---

## Dataset

[CEAS 2008 phishing corpus](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset)
(`CEAS_08.csv`), via Kaggle.

| | Count |
|---|---|
| Total emails | 39,154 |
| Phishing (`label = 1`) | 21,842 |
| Legitimate (`label = 0`) | 17,312 |
| Training split | 31,323 |
| Test split | 7,831 |

Fields used: `body`, `urls`, `label`. The CSV is **not committed** — see
[Running it](#running-it).

---

## What Part 1 found

### Email length — the strongest simple signal

Phishing averages **801 characters** against **2,542** for legitimate mail, which in this corpus
tends to be threaded technical conversation with real substance.

### Urgency keywords — did not hold

Counting `urgent`, `verify`, `verification`, `account`, `password`, `click`, `suspend`,
`unauthorized`, `immediately`, `warning`, `required`, `expiration`.

Legitimate emails scored **higher** on urgency words than phishing did. This corpus skews toward
spam-style phishing that does not lean on urgency language, so the keyword heuristic — the intuitive
choice, and the one the phishing literature emphasizes — is a weak signal here.

Reported rather than quietly dropped: a feature that fails is still a finding, and it is the reason
the modeling stage went to TF-IDF over the full vocabulary instead of a hand-picked keyword list.

### URL counts — weak

Average URL counts were compared across both classes and did not separate them strongly enough to
carry the model on their own.

### Word frequency — useful, once stopwords were filtered

Raw counts were dominated by stopwords, so a filtered pass was added.

- **Phishing:** `news`, `cnncom`, `network`, `settings`, `cnn`, `top`, `cable`, `videos`, `daily`,
  `stories`, `replica`, `alert`
- **Legitimate:** professional vocabulary from genuine threads — `submission`, `added`, `sender`

Overlap between the lists is why the models weight the full vocabulary rather than checking for the
presence of chosen terms.

---

## Method

**Preprocessing.** Null removal, binary label conversion, lowercasing, HTML tag removal, special
character stripping.

**Features.** TF-IDF with `max_features=5000` and English stopword removal for the traditional
models. For BERT, the `bert-base-uncased` tokenizer at `max_length=256`.

**Models.**

| | Configuration |
|---|---|
| Logistic Regression | `max_iter=1000`, on TF-IDF vectors |
| Random Forest | `n_estimators=100`, `random_state=42`, on TF-IDF vectors |
| BERT | `bert-base-uncased`, 3 epochs, lr `2e-5`, batch size 16, AdamW, T4 GPU |

**Evaluation.** Precision, recall, F1, and confusion matrices, on the same 7,831-email test set for
all three models.

---

## Running it

Both notebooks were written in Google Colab and use Colab-specific calls
(`google.colab.files.upload`, `google.colab.drive.mount`). Part 2 additionally needs a **GPU
runtime** for BERT — in Colab, *Runtime → Change runtime type → T4 GPU*.

### Part 1

[![Open Part 1 In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/blkmonday/AI-Phishing-Detection/blob/main/AI_Phishing_Detection.ipynb)

1. Download `CEAS_08.csv` from the Kaggle link above.
2. Run the upload cell and select the CSV.
3. Run the remaining cells. The last one writes `phishing_cleaned.csv` to
   `MyDrive/Phishing_Detection_Projection/`.

### Part 2

[![Open Part 2 In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/blkmonday/AI-Phishing-Detection/blob/main/Phishing%20detection%20Part%202.ipynb)

1. Switch the runtime to a T4 GPU.
2. Run in order — it mounts Drive and reads the `phishing_cleaned.csv` Part 1 produced.

Expect roughly 5 seconds for Logistic Regression, 2 minutes for Random Forest, and 30 minutes for
BERT fine-tuning.

### Locally

Replace the Colab upload cell with `pd.read_csv('CEAS_08.csv')` and skip the Drive-mount cells.
Requires `pandas`, `matplotlib`, `scikit-learn`, and for Part 2, `torch` and `transformers` with a
CUDA GPU.

---

## Future work

- Test against modern phishing corpora — CEAS-08 is from 2008 and its spam-style phishing is
  easier than what TF-IDF would face today
- Evaluate lighter transformers (DistilBERT, TinyBERT) to see whether BERT's precision survives at
  a fraction of the training cost
- Test against AI-generated phishing, where the length and vocabulary signals above likely collapse
- Expand features to header analysis, URL inspection, and attachment scanning
- Integrate into a live filtering pipeline

---

## Related

- [Building a Virtual HomeLab on macOS](https://github.com/blkmonday/Building-A-Virtual-HomeLab-on-MacOS)
  — Kali + Ubuntu lab for offensive and defensive security practice
- [Cybersecurity Projects](https://github.com/blkmonday/Cybersecurity-projects) — labs, writeups, and notes
