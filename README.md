# Leveraing Checkpoints for Membership Inference Attacks

This repository contains the code, configurations, and analysis scripts for my Master's thesis:  
**"Leveraging Model Checkpoints to Improve Membership Inference Attacks in Large Language Models"**.

## 📜 Overview
The project investigates how training checkpoints can be exploited to improve the effectiveness of membership inference attacks (MIAs) against large language models (LLMs). We evaluate across three model families — **CroissantLLM**, **Pythia**, and **OLMo** — and multiple datasets, including **Copyright Traps**, **The Pile**, **Dolma**, and **Paloma**.

We implement and evaluate:
- **Loss trace** and **EMA trace**-based MIAs
- **Attention-based RNN** classifiers
- **Cumulative Loss Fluctuation Average (CLFA)** from the SeqMIA paper
- **Distribution shift detection** (custom BoW + Blind MIA techniques)

## 🚀 Getting Started

### 1️⃣ Clone the repository
```bash
git clone https://github.com/ongsici/checkpoints_mia.git
cd checkpoints_mia
