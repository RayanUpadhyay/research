# research.rayanupadhyay.com

Interactive companions to my two papers. A switcher at the top of the page moves between them.

1. *Fairness and Dependency Preservation in Synthetic Credit Data* (NHSJS Reports, The National High School Journal of Science, 2026), which won a Gold CREST Award, plus a **deluxe edition** of new follow-up experiments. This is the default view.
2. *Emotion-associated representations in the feed forward gates of a language model* (Journal of High School Science, 10(4), 2026, [doi.org/10.64336/001c.175619](https://doi.org/10.64336/001c.175619)), at [research.rayanupadhyay.com/#emotions](https://research.rayanupadhyay.com/#emotions).

**Live:** [research.rayanupadhyay.com](https://research.rayanupadhyay.com)
**Paper:** [nhsjs.com](https://nhsjs.com/2026/fairness-and-dependency-preservation-in-synthetic-credit-data/) · [PDF](https://nhsjs.com/wp-content/uploads/2026/07/Fairness-and-Dependency-Preservation-in-Synthetic-Credit-Data.pdf)

## Paper 2: Emotions at the gate

Everything on this view is taken from the published paper:

- The three levels of explanation (representation, reconfiguration, causal control), what each one found, and the full abstract
- The method, from the 15,895 Chinese texts to the corrected signature neurons in Qwen2.5-1.5B-Instruct's feed-forward gates
- An interactive heatmap of signature neurons per emotion and layer (layers 14–22), with Chinese and English totals
- An overlap wheel of the six emotions, for the Chinese originals and the English translations, with the circumplex-adjacent pairs highlighted and the correlation statistics
- Exclusivity, split-half reliability and classifier F1 per emotion
- The four preregistered task families and what each one showed
- The fourteen hypotheses under the preregistered measure and the two post hoc corrected measures (2, 10 and 11 supported)
- The limitations, the causal experiments that should follow, and a one-click citation

## Paper 1: Features

- The paper in short: the problem, what I did and what I found, plus the full abstract
- A results explorer: pick Baseline, SMOTE, CTGAN, TVAE or Gaussian Copula and see the published ROC AUC, Equalized Odds, Disparate Impact, Predictive Parity and Dependency Preservation Score, with error bars and the other methods for comparison
- Slider labs for Disparate Impact and Predictive Parity, with the paper's results marked on the same scale
- A live heatmap of the real relationships between the dataset's numeric columns
- The pipeline, the limitations and a one-click citation
- **Deluxe edition**: eight new experiments on the questions the paper left open (below)
- Light and dark mode, and a layout that works on phones

## The deluxe edition

New runs, clearly separated from the published results and not peer reviewed. 20 seeds per method, the same train/test splits for every method:

1. **Reproduction** of all five datasets, including TVAE and Gaussian Copula with a current SDV version
2. **A class-balanced baseline** (class weighting and random oversampling), the fairer comparison the paper calls for
3. **Disparate Impact** by method, for women vs men
4. **The synthetic slider**: what happens as synthetic rows replace 0–100% of the real training data
5. **The paper's recommendation**: original data with reweighing, a fairness-aware preprocessing method
6. **Significance tests**: paired Wilcoxon tests against the baseline
7. **Dependencies, re-measured**: our implementation of the Dependency Preservation Score, plus the Frobenius distance
8. **A first privacy check**: distance from synthetic rows to the nearest real applicant

The page states every definition the experiments use, because the paper leaves some details open. The absolute numbers are therefore not directly comparable with the paper's.

## Where the numbers come from

- **Published results, definitions and quotes:** taken from the paper, unchanged.
- **Heatmap and class counts:** computed in the browser from the German Credit data embedded in the page.
- **Deluxe edition:** produced by `experiments/experiments.py`. The summary is embedded in the page; the raw per-run results are in `experiments/results/raw.jsonl`.

To re-run the experiments:

```
pip install -r experiments/requirements.txt
python experiments/experiments.py 20
```

## Data and licence

Hofmann, H. (1994). *Statlog (German Credit Data)*. UCI Machine Learning Repository. https://doi.org/10.24432/C5NC77

The dataset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The copy in `experiments/data/` comes from a public CSV version with readable category labels ([github.com/selva86/datasets](https://github.com/selva86/datasets)); the rows are the same as the UCI original. Some of the readable labels in that copy don't exactly match UCI's own codes (the paper also notes known label problems in this dataset).

## Tech

The page is a single `index.html` with plain HTML, CSS and JavaScript, no libraries and no build step needed to host it. Fonts come from Google Fonts. The experiments use Python with scikit-learn, XGBoost, imbalanced-learn and SDV. Hosted on Vercel.
