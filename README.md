# Leveraging Checkpoints for Membership Inference Attacks

This repository contains the code, configurations, and analysis scripts for my Master's thesis:  
**"Leveraging Model Checkpoints to Improve Membership Inference Attacks in Large Language Models"**.

## Abstract

Membership Inference Attacks (MIAs) are commonly employed to evaluate privacy risks in machine learning models, providing insights on potential memorisation that could lead to data leakage. However, state-of-the-art MIA techniques require training a large number of shadow models with an architecture similar to that of the target model, making them computationally intensive and impractical for iterative use during model development, particularly for large models. Alternative threshold-based attacks that rely on confidence-based signals have suboptimal performance, and only provides a lower-bound of the model's privacy risks. 

In this work, we investigate an alternative method that leverages intermediate model checkpoints, an artifact of pre-training Large Language Models (LLMs). We hypothesise that these checkpoints embed important temporal information on model behaviour that can be exploited to improve MIA performance with significant computational savings compared to shadow model MIAs. To ensure a bias-free evaluation, we first extensively evaluate sampled datasets to ensure a clean setup, free from distribution shifts. We then propose and evaluate several checkpoint MIA methods, among which two demonstrate strong performance: (i) Cumulative Loss Fluctuation Amplititude (CLFA) [(Li et al.)](https://arxiv.org/pdf/2407.15098) which measures the amplitude of loss variation across training steps, and (ii) Steps to Convergence (S2Conv), which measures the number of steps required for the loss to first fall within a defined margine of the final loss values. When applied to normalised loss traces (sequence of sample losses over training steps) evaluated on 144 checkpoints from Pythia 6.9B, CLFA achieved an Area Under Curve (AUC) of 0.836 on Wikipedia subset of The Pile with sequence length of 2048, and S2Conv attained a 33.4\% True Positive Rate (TPR) @ 1\% False Positive Rate (FPR) on the same dataset. We estimate theoretically, that the use of checkpoint MIAs provide computational savings of at least 66\% (measured in GPU-hours) as compared to shadow model MIAs. Ablation and sensitvity studies on checkpoint granularity and model size reveal that MIA performance increases with number of checkpoints used, and with increasing model size. These findings demonstrate that intermediate model checkpoints contain important temporal dynamics on the model's behaviour and can be leveraged by model developers as an auditing tool for assessing privacy risks in LLMs.



## 📜 Overview
The aim of this project is to leverage intermediate model checkpoints from pre-training of LLMs to develop an MIA method that is less computationally intensive and more scalable than shadow model attacks, while still outperforming simple threshold-based attacks. 

Our contributions are as follows:
- Developed a clean benchmark setup for MIA evaluation consisting of 2,000 samples (1,000 members and 1,000 non-members) with 6 sequence lengths (64, 128, 256, 512, 1024 and 2048) for each of the 5 datasets from The Pile - Wikipedia, PubMed Central, USPTO, Pile-CC and GitHub. Rigorous validation procedures were applied to ensure that the setup was free from distribution shifts.
- Applied reservoir sampling as a computational efficient method to randomly sample the train and test split of The Pile. 
- Proposed and systematically evaluated several checkpoint MIA methods. Among these, two methods demonstrated strong performance: (i) CLFA, which achieved the highest AUC and (ii) S2Conv, which achieved the highest TPR @ 1\% FPR. Both methods outperformed baseline MIA methods. 
- Theoretical estimate on computational savings with the use of checkpoint MIA methods as compared to shadow model attacks.
- Conducted ablation and sensitivity studies on factors affecting MIA performance such as model size, checkpoint granularity and sequence lengths. These findings provide valuable insights for model auditors to understand performance trade-offs based on those factors. 


## 🚀 Getting Started

### Key Components of Repository

1. `distribution_shift_detection` 

This folder contains the codes and notebooks for the three methods employed to validate that our setup is clean and free from distribution shift: (i) Random Forest Bag-of-words Classifier, (ii) PCA and t-SNE visualisation of sample distributions and (iii) adapted [Blind MIA](https://github.com/ethz-spylab/Blind-MIA) 

2. `checkpoint_mia_techniques`

This folder contains the bulk of the work done:
- Implemention of proposed checkpoint MIA methods
- Notebooks that evaluate MIA methods on Pythia and OLMo models 
- Folders containing MIA results from evaluations and notebooks used to generate results plots

3. `reservoir_sampling`

This folder contains the code that was used to perform reservoir sampling for [The Pile](https://huggingface.co/datasets/monology/pile-uncopyrighted) and [Dolma](https://huggingface.co/datasets/allenai/dolma) datasets. 

We share the sampled datasets on HuggingFace: (1) [Pile-RS-Truncated](https://huggingface.co/datasets/ongsici/Pile-RS-Truncated) , (2) [Pile-RS-Independent](https://huggingface.co/datasets/ongsici/Pile-RS-Independent) and (3) [Dolma and Paloma](https://huggingface.co/datasets/ongsici/Dolma-Paloma)

4. `data`

We separately share the pre-processed formats of the sampled data, and results of checkpoint losses evaluated for all samples. The data can be downloaded [here](https://imperiallondon-my.sharepoint.com/:f:/g/personal/sco115_ic_ac_uk/EvLGD6RHBsFLo91TbeOKrMIBNEpT-4emQKPFVR6FXvAGUg?e=AeqMLd) and placed within this folder.

<!-- ### 1️⃣ Clone the repository
```bash
git clone https://github.com/ongsici/checkpoints_mia.git
cd checkpoints_mia -->
