# research.rayanupadhyay.com

An interactive companion to my paper, *Fairness and Dependency Preservation in Synthetic Credit Data* (NHSJS Reports, The National High School Journal of Science, 2026), which won a Gold CREST Award.

**Live:** [research.rayanupadhyay.com](https://research.rayanupadhyay.com)
**Paper:** [nhsjs.com](https://nhsjs.com/2026/fairness-and-dependency-preservation-in-synthetic-credit-data/) · [PDF](https://nhsjs.com/wp-content/uploads/2026/07/Fairness-and-Dependency-Preservation-in-Synthetic-Credit-Data.pdf)

## Features

- The short version: the problem, what I did and what I found, plus the full abstract
- A results explorer: pick Baseline, SMOTE, CTGAN, TVAE or Gaussian Copula and see its ROC AUC, Equalized Odds, Disparate Impact, Predictive Parity and Dependency Preservation Score on scales that show the other methods too, with the error bars from 10 runs
- Why TVAE and Gaussian Copula show NaN, in the paper's own words
- Slider labs for Disparate Impact and Predictive Parity: set two groups' rates and watch the metric change, with the paper's results marked on the same scale
- A heatmap of the real relationships between the dataset's numeric columns (Spearman rank correlations), computed live from all 1,000 applicants
- How the Dependency Preservation Score is defined, the class balance before and after resampling, the pipeline, and the limitations
- A one-click citation
- Light and dark mode, and a layout that works on phones

## Where the numbers come from

Every result, definition and quote is taken from the published paper. Nothing is estimated or simulated. The only things computed on the page are:

- the correlation heatmap, which is calculated from the German Credit data embedded in the page (the seven numeric columns only)
- the class counts (700 good, 300 bad)
- whatever you set with the sliders

The data embedded in the page comes from a public CSV copy of the German Credit dataset (UCI Statlog); only the numeric columns and the credit label are included.

## Tech

A single `index.html` with plain HTML, CSS and JavaScript, no libraries and no build step. Fonts come from Google Fonts. Hosted on Vercel.
