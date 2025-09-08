## Distribution Shift Detection

We used three methods to evaluate for presence of distribution shifts in the sampled datasets:
1. Random Forest Bag-of-words classifier
2. Distribution visualisation using PCA and t-SNE
3. [Blind MIA](https://github.com/ethz-spylab/Blind-MIA) methods


Both #1 and #2 can be run indepdently using the python notebook, `1_bag_of_words_classifier.ipynb` and `2_pca_tsen_visualisation.ipynb` respectively.

### Blind MIA

We adapted blind-MIA from Das et al. to be run using the python notebook within `3_blind_mia` folder. This included:
- Date detection: scanning through a range of dates from 1990 to 2024 (inclusive), and reporting the max AUC and max TPR scores.
- Greedy rare word selection: adapted to report AUC as well, on top of TPR scores