<div align="center">

<a href="https://causalml-book.org">
  <img src="https://causalml-book.org/assets/logo-dark.png" alt="CausalML Book" width="520">
</a>

<br><br>

<a href="https://causalml-book.org">
  <img src="https://causalml-book.org/assets/metaimage.png" alt="Applied Causal Inference Powered by ML and AI" width="700">
</a>

<br><br>

[![Website](https://img.shields.io/badge/CausalML--Book.org-Visit_Website-0a1f6f?style=for-the-badge&logo=google-chrome&logoColor=white)](https://causalml-book.org)
&nbsp;
[![arXiv](https://img.shields.io/badge/arXiv-2403.02467-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white)](https://doi.org/10.48550/arXiv.2403.02467)
&nbsp;
[![Labs](https://img.shields.io/badge/Interactive_Labs-Open-00A99D?style=for-the-badge&logo=jupyter&logoColor=white)](https://causalml-book.org/labs)

<br>

[![Notebooks](https://img.shields.io/badge/Notebooks_Repo-MetricsMLNotebooks-181717?style=flat-square&logo=github)](https://github.com/CausalAIBook/MetricsMLNotebooks)

---

**A comprehensive, rigorous guide to modern causal inference methods powered by machine learning.**

*By* &ensp;
[**Victor Chernozhukov**](https://www.victorchernozhukov.com)<sup>1</sup> &ensp;&bull;&ensp;
[**Christian Hansen**](https://voices.uchicago.edu/christianhansen/)<sup>2</sup> &ensp;&bull;&ensp;
[**Nathan Kallus**](https://nathankallus.com/)<sup>3</sup> &ensp;&bull;&ensp;
[**Martin Spindler**](https://www.bwl.uni-hamburg.de/en/statistik/team/spindler.html)<sup>4</sup> &ensp;&bull;&ensp;
[**Vasilis Syrgkanis**](https://vsyrgkanis.com/)<sup>5</sup>

<sup>1</sup>Massachusetts Institute of Technology &ensp;
<sup>2</sup>University of Chicago &ensp;
<sup>3</sup>Cornell University &ensp;
<sup>4</sup>Universität Hamburg &ensp;
<sup>5</sup>Stanford University

</div>

---

## About the Book

This book bridges the gap between **machine learning** and **causal inference**, providing rigorous methods for answering causal questions using modern ML tools. Topics span predictive inference, causal identification, double/debiased machine learning, heterogeneous treatment effects, instrumental variables, difference-in-differences, regression discontinuity, and more.

> **Read the full book and download individual chapters at [CausalML-Book.org](https://causalml-book.org)**

---

## Chapters

All chapters are available for free at **[causalml-book.org](https://causalml-book.org)**.

### Preamble

| | Chapter |
|:---:|---------|
| P | Preface |
| 0 | Powering Causal Inference with ML and AI |

### Core Material

| | Chapter | Topics |
|:---:|---------|--------|
| 1 | Predictive Inference with Linear Regression in Moderately High Dimensions | `Prediction` `Inference` |
| 2 | Causal Inference via Randomized Experiments | `Causality` `Inference` |
| 3 | Predictive Inference via Modern High-Dimensional Linear Regression | `Prediction` |
| 4 | Statistical Inference on Predictive Effects in High-Dimensional Linear Regression Models | `Causality` `Inference` |
| 5 | Causal Inference via Conditional Ignorability | `Causality` |
| 6 | Causal Inference via Linear Structural Equations | `Causality` |
| 7 | Causal Inference via DAGs and Nonlinear Structural Equation Models | `Causality` |
| 8 | Predictive Inference via Modern Nonlinear Regression | `Prediction` |
| 9 | Statistical Inference on Predictive and Causal Effects in Modern Nonlinear Regression Models | `Causality` `Inference` |
| 10 | Feature Engineering for Causal and Predictive Inference | `Causality` `Inference` |

### Advanced Topics

| | Chapter |
|:---:|---------|
| 11 | Deeper Dive into DAGs, Good and Bad Controls |
| 12 | Unobserved Confounders, Instrumental Variables, and Proxy Controls |
| 13 | DML for IV and Proxy Controls Models and Robust DML Inference under Weak Identification |
| 14 | Statistical Inference on Heterogeneous Treatment Effects |
| 15 | Estimation and Validation of Heterogeneous Treatment Effects |
| 16 | Difference-in-Differences |
| 17 | Regression Discontinuity Designs |

---

## Interactive Labs

<div align="center">

All labs run directly in **Google Colab** — no local setup required.

[![Browse All Labs](https://img.shields.io/badge/Browse_All_Labs-CausalML--Book.org%2Flabs-0a1f6f?style=for-the-badge&logo=google-chrome&logoColor=white)](https://causalml-book.org/labs)
&nbsp;
[![Notebooks Repo](https://img.shields.io/badge/Source_Code-MetricsMLNotebooks-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/CausalAIBook/MetricsMLNotebooks)

<br>

<img src="https://causalml-book.org/assets/python.svg" height="28" alt="Python">&ensp;
<img src="https://causalml-book.org/assets/r.svg" height="28" alt="R">&ensp;
Available in **Python** and **R**

</div>

<br>

<details>
<summary>&ensp;<b>Ch 1 &mdash; Prediction & Linear Regression</b></summary>
<br>

| Lab | Python | R |
|-----|:------:|:-:|
| OLS and Lasso for Wage Prediction | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CausalAIBook/MetricsMLNotebooks/blob/main/PM1/python-ols-and-lasso-for-wage-prediction.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CausalAIBook/MetricsMLNotebooks/blob/main/PM1/r-ols-and-lasso-for-wage-prediction.irnb) |
| The Gender Wage Gap | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CausalAIBook/MetricsMLNotebooks/blob/main/PM1/python-ols-and-lasso-for-wage-gap-inference.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CausalAIBook/MetricsMLNotebooks/blob/main/PM1/r-ols-and-lasso-for-wage-gap-inference.irnb) |
| Exercise on Overfitting | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CausalAIBook/MetricsMLNotebooks/blob/main/PM1/python-linear-model-overfitting.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CausalAIBook/MetricsMLNotebooks/blob/main/PM1/r-linear-model-overfitting.irnb) |

</details>

<details>
<summary>&ensp;<b>Ch 2 &mdash; Randomized Experiments</b></summary>
<br>

| Lab | Python | R |
|-----|:------:|:-:|
| Vaccination RCT (Polio 1954) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CausalAIBook/MetricsMLNotebooks/blob/main/CM1/python-rct-vaccines.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CausalAIBook/MetricsMLNotebooks/blob/main/CM1/r-rct-vaccines.irnb) |
| Covariates in RCT: Precision Adjustment | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CausalAIBook/MetricsMLNotebooks/blob/main/CM1/python-sim-precision-adj.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CausalAIBook/MetricsMLNotebooks/blob/main/CM1/r-sim-precision-adj.irnb) |
| Reemployment Bonus RCT | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CausalAIBook/MetricsMLNotebooks/blob/main/CM1/python-rct-penn-precision-adj.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CausalAIBook/MetricsMLNotebooks/blob/main/CM1/r-rct-penn-precision-adj.irnb) |

</details>

<details>
<summary>&ensp;<b>Ch 3 &mdash; High-Dimensional Linear Regression</b></summary>
<br>

| Lab | Python | R |
|-----|:------:|:-:|
| Penalized Linear Regressions: Simulation | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CausalAIBook/MetricsMLNotebooks/blob/main/PM2/python_linear_penalized_regs.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CausalAIBook/MetricsMLNotebooks/blob/main/PM2/r_linear_penalized_regs.irnb) |
| Case Study: Wage Prediction with ML | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CausalAIBook/MetricsMLNotebooks/blob/main/PM2/python_ml_for_wage_prediction.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CausalAIBook/MetricsMLNotebooks/blob/main/PM2/r_ml_for_wage_prediction.irnb) |

</details>

<details>
<summary>&ensp;<b>Ch 4 &mdash; Inference in High-Dimensional Models</b></summary>
<br>

| Lab | Python | R |
|-----|:------:|:-:|
| Simulation on Orthogonal Estimation | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CausalAIBook/MetricsMLNotebooks/blob/main/PM2/python_orthogonal_orig.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CausalAIBook/MetricsMLNotebooks/blob/main/PM2/r_orthogonal_orig.irnb) |
| Comparing Orthogonal vs Non-Orthogonal Methods | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CausalAIBook/MetricsMLNotebooks/blob/main/PM2/python_experiment_non_orthogonal.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CausalAIBook/MetricsMLNotebooks/blob/main/PM2/r_experiment_non_orthogonal.irnb) |
| Testing the Convergence Hypothesis | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CausalAIBook/MetricsMLNotebooks/blob/main/PM2/python_convergence_hypothesis_double_lasso.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CausalAIBook/MetricsMLNotebooks/blob/main/PM2/r_convergence_hypothesis_double_lasso.irnb) |
| Heterogeneous Effect of Sex on Wage | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CausalAIBook/MetricsMLNotebooks/blob/main/PM2/python_heterogeneous_wage_effects.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CausalAIBook/MetricsMLNotebooks/blob/main/PM2/r_heterogenous_wage_effects.irnb) |

</details>

<details>
<summary>&ensp;<b>Ch 6–7 &mdash; DAGs & Structural Equations</b></summary>
<br>

| Lab | Python | R |
|-----|:------:|:-:|
| Collider Bias (Hollywood) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CausalAIBook/MetricsMLNotebooks/blob/main/CM2/python-colliderbias-hollywood.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CausalAIBook/MetricsMLNotebooks/blob/main/CM2/r-colliderbias-hollywood.irnb) |
| Causal Identification in DAGs | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CausalAIBook/MetricsMLNotebooks/blob/main/CM3/python-pgmpy.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CausalAIBook/MetricsMLNotebooks/blob/main/CM3/r-dagitty.irnb) |
| DoSearch for Causal Identification | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CausalAIBook/MetricsMLNotebooks/blob/main/CM3/python-dosearch.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CausalAIBook/MetricsMLNotebooks/blob/main/CM3/r-dosearch.irnb) |

</details>

<details>
<summary>&ensp;<b>Ch 8 &mdash; Nonlinear Prediction</b></summary>
<br>

| Lab | Python | R |
|-----|:------:|:-:|
| ML Estimators for Wage Prediction | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CausalAIBook/MetricsMLNotebooks/blob/main/PM3/python-nonlinear-ml-for-wage-prediction.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CausalAIBook/MetricsMLNotebooks/blob/main/PM3/r_ml_wage_prediction.irnb) |
| Functional Approximations by Trees and Neural Nets | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CausalAIBook/MetricsMLNotebooks/blob/main/PM3/python-functional-approximation-by-nn-and-rf.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CausalAIBook/MetricsMLNotebooks/blob/main/PM3/r_functional_approximation_by_nn_and_rf.irnb) |

</details>

<details>
<summary>&ensp;<b>Ch 9 &mdash; DML for Causal & Predictive Effects</b></summary>
<br>

| Lab | Python | R |
|-----|:------:|:-:|
| Effect of Gun Ownership on Homicide | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CausalAIBook/MetricsMLNotebooks/blob/main/PM4/python_dml_inference_for_gun_ownership.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CausalAIBook/MetricsMLNotebooks/blob/main/PM4/r_dml_inference_for_gun_ownership.irnb) |
| DAG Analysis of 401(k) Impact | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CausalAIBook/MetricsMLNotebooks/blob/main/PM4/python-identification-analysis-of-401-k-example-w-dags.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CausalAIBook/MetricsMLNotebooks/blob/main/PM4/r-identification-analysis-of-401-k-example-w-dags.irnb) |
| DML Inference on 401(k) Wealth Effects | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CausalAIBook/MetricsMLNotebooks/blob/main/PM4/python-dml-401k.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CausalAIBook/MetricsMLNotebooks/blob/main/PM4/r-dml-401k.irnb) |
| DML for Partially Linear Model (Growth) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CausalAIBook/MetricsMLNotebooks/blob/main/PM4/python_debiased_ml_for_partially_linear_model_growth.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CausalAIBook/MetricsMLNotebooks/blob/main/PM4/r_debiased_ml_for_partially_linear_model_growth.irnb) |

</details>

<details>
<summary>&ensp;<b>Ch 10 &mdash; Feature Engineering</b></summary>
<br>

| Lab | Python | R |
|-----|:------:|:-:|
| Variational Autoencoders and PCA | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CausalAIBook/MetricsMLNotebooks/blob/main/PM5/Autoencoders.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CausalAIBook/MetricsMLNotebooks/blob/main/PM5/Autoencoders.irnb) |
| DoubleML Feature Engineering with BERT | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CausalAIBook/MetricsMLNotebooks/blob/main/PM5/DoubleML_and_Feature_Engineering_with_BERT.ipynb) | &mdash; |

</details>

<details>
<summary>&ensp;<b>Ch 12–13 &mdash; IV, Proxy Controls & Weak Identification</b></summary>
<br>

| Lab | Python | R |
|-----|:------:|:-:|
| Sensitivity Analysis with Sensmakr | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CausalAIBook/MetricsMLNotebooks/blob/main/AC1/python-sensitivity-analysis-with-sensmakr-and-debiased-ml.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CausalAIBook/MetricsMLNotebooks/blob/main/AC1/r-sensitivity-analysis-with-sensmakr-and-debiased-ml.irnb) |
| Negative (Proxy) Controls | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CausalAIBook/MetricsMLNotebooks/blob/main/AC1/python-proxy-controls.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CausalAIBook/MetricsMLNotebooks/blob/main/AC1/r-proxy-controls.irnb) |
| DML for 401(k) with IV | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CausalAIBook/MetricsMLNotebooks/blob/main/AC2/python-dml-401k-IV.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CausalAIBook/MetricsMLNotebooks/blob/main/AC2/r-dml-401k-IV.irnb) |
| Weak IV Experiments | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CausalAIBook/MetricsMLNotebooks/blob/main/AC2/python-weak-iv-experiments.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CausalAIBook/MetricsMLNotebooks/blob/main/AC2/r-weak-iv-experiments.irnb) |
| DML for Partially Linear IV Model | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CausalAIBook/MetricsMLNotebooks/blob/main/AC2/python-debiased-ml-for-partially-linear-iv-model.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CausalAIBook/MetricsMLNotebooks/blob/main/AC2/r-debiased-ml-for-partially-linear-iv-model.irnb) |

</details>

<details>
<summary>&ensp;<b>Ch 14–16 &mdash; Heterogeneous Effects & Diff-in-Diff</b></summary>
<br>

| Lab | Python | R |
|-----|:------:|:-:|
| CATE Estimation with Causal Forests | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CausalAIBook/MetricsMLNotebooks/blob/main/T/CATE-estimation.ipynb) | &mdash; |
| CATE Inference: Best Linear Predictors | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CausalAIBook/MetricsMLNotebooks/blob/main/T/CATE-inference.ipynb) | &mdash; |
| Conditional Average Treatment Effects | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CausalAIBook/MetricsMLNotebooks/blob/main/T/dml-for-conditional-average-treatment-effect.irnb) | &mdash; |
| Difference-in-Differences: Minimum Wage | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CausalAIBook/MetricsMLNotebooks/blob/main/T/T-3%20Diff-in-Diff%20Minimum%20Wage%20Example.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CausalAIBook/MetricsMLNotebooks/blob/main/T/T-3%20Diff-in-Diff%20Minimum%20Wage%20Example.irnb) |

</details>

---

## Software Packages

Companion open-source implementations for the methods covered in the book.

<table>
<tr>
<td align="center" width="33%">

<a href="https://docs.doubleml.org/stable/">
  <img src="https://docs.doubleml.org/dev/logo.png" alt="DoubleML" width="160"><br>
  <b>DoubleML</b>
</a>

Double/Debiased ML in Python & R

[![Docs](https://img.shields.io/badge/Docs-blue?style=flat-square)](https://docs.doubleml.org/stable/)
[![Python](https://img.shields.io/badge/GitHub-Python-181717?style=flat-square&logo=github)](https://github.com/DoubleML/doubleml-for-py)
[![R](https://img.shields.io/badge/GitHub-R-181717?style=flat-square&logo=github)](https://github.com/DoubleML/doubleml-for-r)

</td>
<td align="center" width="33%">

<a href="https://econml.azurewebsites.net/">
  <img src="https://econml.azurewebsites.net/_static/econml-logo-inverse.png" alt="EconML" width="160"><br>
  <b>EconML</b>
</a>

Heterogeneous Treatment Effects

[![Docs](https://img.shields.io/badge/Docs-blue?style=flat-square)](https://econml.azurewebsites.net/)
[![GitHub](https://img.shields.io/badge/GitHub-EconML-181717?style=flat-square&logo=github)](https://github.com/py-why/EconML)

</td>
<td align="center" width="33%">

<a href="https://statalasso.github.io/">
  <b>Stata ML & ddml</b>
</a>

<br>

Regularized Regression & DML for Stata

[![Stata ML](https://img.shields.io/badge/Docs-statalasso-blue?style=flat-square)](https://statalasso.github.io/)
[![ddml R](https://img.shields.io/badge/ddml-R_Package-blue?style=flat-square)](https://thomaswiemann.com/ddml/)

</td>
</tr>
<tr>
<td>

PLR, IRM, PLIV, IIVM models. Builds on [scikit-learn](https://scikit-learn.org/) (Python) and [mlr3](https://mlr3.mlr-org.com/) (R).

</td>
<td>

Double ML, Causal Forests, Meta-Learners, IV methods. Part of [PyWhy](https://www.pywhy.org/).

</td>
<td>

[lassopack](https://statalasso.github.io/docs/lassopack/), [pdslasso](https://statalasso.github.io/docs/pdslasso/), [pystacked](https://statalasso.github.io/docs/pystacked/), [ddml](https://statalasso.github.io/docs/ddml/).

</td>
</tr>
</table>

---

## Citation

```bibtex
@article{chernozhukov2024applied,
  title   = {Applied Causal Inference Powered by ML and AI},
  author  = {Chernozhukov, Victor and Hansen, Christian and Kallus, Nathan
             and Spindler, Martin and Syrgkanis, Vasilis},
  journal = {arXiv preprint arXiv:2403.02467},
  year    = {2024},
  doi     = {10.48550/arXiv.2403.02467}
}
```

---

<div align="center">

<a href="https://causalml-book.org">
  <img src="https://causalml-book.org/assets/logo-dark.png" alt="CausalML Book" width="280">
</a>

<br><br>

[![Website](https://img.shields.io/badge/Website-CausalML--Book.org-0a1f6f?style=flat-square&logo=google-chrome&logoColor=white)](https://causalml-book.org)
&ensp;
[![Labs](https://img.shields.io/badge/Labs-Interactive_Notebooks-00A99D?style=flat-square&logo=jupyter&logoColor=white)](https://causalml-book.org/labs)
&ensp;
[![Notebooks](https://img.shields.io/badge/GitHub-MetricsMLNotebooks-181717?style=flat-square&logo=github)](https://github.com/CausalAIBook/MetricsMLNotebooks)
&ensp;
[![arXiv](https://img.shields.io/badge/arXiv-2403.02467-b31b1b?style=flat-square&logo=arxiv)](https://doi.org/10.48550/arXiv.2403.02467)

</div>
