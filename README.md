# MLOps-Micro-Cohort
# 🧬 PathwayMLOps4RareDx
**A Paired-Sample, Pathway-Anchored MLOps Framework for Robust Transcriptome Classification in Small Cohorts**

This repository accompanies the publication:

> Shabanian et al., *Paired-Sample and Pathway-Anchored MLOps Framework for Robust Transcriptomic Machine Learning in Small Cohorts* (JMIR Bioinformatics, 2025)

---

## 🔍 Overview

High-dimensional transcriptomic data + small cohorts = overfitting.  
This project introduces a biologically informed classification pipeline for micro-cohorts (<30 subjects), integrating:

- 🧪 **Paired-sample transcriptomics** (e.g., tumor vs. normal)
- 🧬 **N-of-1 pathway analytics** (Wilcoxon-based GO enrichment per subject)
- ⚙️ **Reproducible MLOps** with [Weights & Biases](https://wandb.ai/) (versioning, sweep tuning, logging)
- 📉 **Retroactive Ablation** analysis to validate feature robustness

---

## 📂 Project Structure

- `notebooks/` – Reproducible Jupyter notebooks (data transformation, training, evaluation)
- `src/` – Core utilities (pathway analysis, RF model, MLOps interface)
- `sweep/` – YAML configs for W&B sweeps
- `results/` – Figures, tables, and exported W&B outputs

---

## 📊 Case Studies

| Dataset | Condition         | Accuracy | Precision | Recall |
|---------|-------------------|----------|-----------|--------|
| HRV     | Symptomatic vs. Asymptomatic | 95%      | 97%       | 95%    |
| BC      | TP53 vs. PIK3CA   | 89%      | 90%       | 90%    |

---

## 🚀 How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Launch training
python src/train_model.py --config sweep/sweep.yaml

