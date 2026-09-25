# AI Phishing Email Detection

Exploratory analysis and feature engineering on a labeled phishing email corpus, working toward a
machine-learning classifier that separates phishing from legitimate mail.

Part of my cybersecurity coursework and portfolio as a Computer Science student at
Southern University A&M College.

---

## Status

**Stage 1 of 2 — data exploration and feature engineering complete.**

The notebook establishes which signals actually separate phishing from legitimate email in this
dataset. No classifier has been trained yet; that is the next step. Findings so far are documented
honestly below, including the features that turned out *not* to work.

---

## Dataset

[CEAS 2008 phishing corpus](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset)
(`CEAS_08.csv`), via Kaggle.

| | Count |
|---|---|
| Total emails | 39,154 |
| Phishing (`label = 1`) | 21,842 |
| Legitimate (`label = 0`) | 17,312 |

Relevant columns: `body`, `urls`, `label`.

The CSV is **not committed** to this repo. See [Running the notebook](#running-the-notebook).

---

## Features explored

### 1. Email length — useful

Phishing emails are markedly shorter. Average phishing body length is roughly 801 characters,
against a much longer average for legitimate mail, which tends to contain threaded technical
conversation with real substance. Overlaid histograms show the distributions separating clearly
below ~5,000 characters.

### 2. Urgency keywords — **not** useful in this corpus

Counting occurrences of `urgent`, `verify`, `verification`, `account`, `password`, `click`,
`suspend`, `unauthorized`, `immediately`, `warning`, `required`, `expiration`.

Contrary to expectation, legitimate emails in this dataset scored *higher* on urgency words than
phishing ones. This corpus skews toward spam-style phishing that does not lean on urgency language,
so keyword presence is a weak signal here. Documented rather than dropped, because the negative
result is the finding.

### 3. URL count — useful

Taken from the dataset's existing `urls` column, comparing average URL counts across both classes.

### 4. Word frequency with stopword filtering — useful

Raw frequency counts were dominated by stopwords, so a filtered pass was added. What surfaced:

- **Phishing:** `news`, `cnncom`, `network`, `settings`, `cnn`, `top`, `cable`, `videos`, `daily`,
  `stories`, `replica`, `alert`
- **Legitimate:** more professional vocabulary from genuine threads — `submission`, `added`, `sender`

Some tokens appear in both classes, so these need weighting rather than presence checks.

---

## Running the notebook

The notebook was written in Google Colab and uses Colab-specific calls
(`google.colab.files.upload`, `google.colab.drive.mount`).

### In Colab (as written)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/blkmonday/AI-Phishing-Detection/blob/main/AI_Phishing_Detection.ipynb)

1. Open the notebook via the badge above.
2. Download `CEAS_08.csv` from the Kaggle link above.
3. Run the upload cell and select the CSV when prompted.
4. Run the remaining cells in order.

The final cell mounts Google Drive and writes `phishing_cleaned.csv` to
`MyDrive/Phishing_Detection_Projection/`.

### Locally

Replace the Colab upload cell with a direct read:

```python
import pandas as pd
df = pd.read_csv('CEAS_08.csv')
```

and skip the Drive-mount cell. Requires `pandas` and `matplotlib`.

---

## Next steps

- [ ] Vectorize email bodies (TF-IDF) and train a baseline classifier
- [ ] Compare logistic regression against Naive Bayes and a tree ensemble
- [ ] Evaluate with precision/recall and a confusion matrix — false negatives matter more than
      overall accuracy for phishing detection
- [ ] Weight word-frequency features rather than testing presence, given the class overlap
- [ ] Validate against a second corpus to test whether the urgency-keyword result generalizes

---

## Related

- [Building a Virtual HomeLab on macOS](https://github.com/blkmonday/Building-A-Virtual-HomeLab-on-MacOS)
  — Kali + Ubuntu lab for offensive and defensive security practice
- [Cybersecurity Projects](https://github.com/blkmonday/Cybersecurity-projects) — labs, writeups, and notes
