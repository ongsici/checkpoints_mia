import numpy as np
# import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
import random
# from ortools.algorithms.python import knapsack_solver
import itertools
from collections import defaultdict
# from datasets import load_dataset

# returns the set of all N-grams, for N≤n
# e.g., if n=3 and s="ababacd", this returns
# {"a", "b", "c", "d", "ab", "ba", "ac", "cd", "aba", "bab", "bac", "acd"}
def ngrams(s, n=5):
    if n == 1:
        return set(s)
    iters = itertools.tee(s, n)                                                     
    for i, it in enumerate(iters):                                               
        next(itertools.islice(it, i, i), None)                                               
    return set("".join(x) for x in zip(*iters)).union(ngrams(s, n=n-1))

# list of sentences for each ngram
def count_unique_chars(lines, threshold=None):
    char_counts = defaultdict(set)
    for i, line in enumerate(lines):
        unique_chars = ngrams(line)
        for char in unique_chars:
            char_counts[char].add(i)
    
    if threshold is not None:
        for c in list(char_counts.keys()):
            if len(char_counts[c]) < threshold:
                del char_counts[c]

    return char_counts


def greedy_selection_basic(member_lines, nonmember_lines, dataset_name, fpr_budget, plot_roc):
    
    TEST_SIZE = 0.1
    CROSS_VALS = 10
    RES = []
    AUCS = [] 
    TARGET_FPR = fpr_budget

    # member_lines = list(dataset['member'])
    # nonmember_lines = list(dataset['nonmember'])
    N = int(TEST_SIZE*len(member_lines))
    # print(N)

    for _ in range(CROSS_VALS):
        member_sample = np.random.permutation(member_lines)
        nonmember_sample = np.random.permutation(nonmember_lines)

        # make a train-test split
        member_test = member_sample[-N:]
        nonmember_test = nonmember_sample[-N:]

        member_sample = member_sample[:-N]
        nonmember_sample = nonmember_sample[:-N]

        # only consider ngrams that appear at least THRESHOLD times in the
        # training set to guard against overfitting
        THRESHOLD = 1

        # Count ngrams in member and nonmember captions (train)
        member_char_counts = count_unique_chars(member_sample, threshold=THRESHOLD)
        nonmember_char_counts = count_unique_chars(nonmember_sample)

        # Count ngrams in member and nonmember captions (test)
        member_char_counts_test = count_unique_chars(member_test)
        nonmember_char_counts_test = count_unique_chars(nonmember_test)

        BUDGET = int(1.0 * len(nonmember_sample))
        CURR_TPR = 0
        CURR_FPR = 0
        CURR_TPR_TEST = 0
        CURR_FPR_TEST = 0
        sol_chars = []

        best_tpr = 0

        while CURR_FPR < BUDGET:
            candidates = [c for c in sorted(member_char_counts.keys()) if len(nonmember_char_counts[c]) <= BUDGET - CURR_FPR]
            if len(candidates) == 0:
                break
            ratios = [(len(member_char_counts[c]) + 0) / (len(nonmember_char_counts[c])+1) for c in candidates]
            best_idx = np.argmax(ratios)
            chosen_c = candidates[best_idx]
            sol_chars.append(chosen_c)

            CURR_TPR += len(member_char_counts[chosen_c])
            CURR_FPR += len(nonmember_char_counts[chosen_c])

            #print("TPR", sorted(list(member_char_counts[chosen_c])))
            #print("FPR", sorted(list(nonmember_char_counts[chosen_c])))

            temp = member_char_counts[chosen_c].copy()
            for c in list(member_char_counts.keys()):
                member_char_counts[c] -= temp
                if len(member_char_counts[c]) == 0:
                    del member_char_counts[c]

            temp = nonmember_char_counts[chosen_c].copy()
            for c in list(nonmember_char_counts.keys()):
                nonmember_char_counts[c] -= temp
                if len(nonmember_char_counts[c]) == 0:
                    del nonmember_char_counts[c]

            CURR_TPR_TEST += len(member_char_counts_test[chosen_c])
            CURR_FPR_TEST += len(nonmember_char_counts_test[chosen_c])

            temp = member_char_counts_test[chosen_c].copy()
            for c in list(member_char_counts_test.keys()):
                member_char_counts_test[c] -= temp
                if len(member_char_counts_test[c]) == 0:
                    del member_char_counts_test[c]

            temp = nonmember_char_counts_test[chosen_c].copy()
            for c in list(nonmember_char_counts_test.keys()):
                nonmember_char_counts_test[c] -= temp
                if len(nonmember_char_counts_test[c]) == 0:
                    del nonmember_char_counts_test[c]

            #print(chosen_c, 
            #      100 * CURR_TPR / len(member_sample), 
            #      100 * CURR_FPR / len(nonmember_sample), 
            #      100 * CURR_TPR_TEST / len(member_test), 
            #      100 * CURR_FPR_TEST / len(nonmember_test))

            m = [i for i,s in enumerate(member_sample) if len(ngrams(s).intersection(sol_chars)) > 0]
            nm = [s for i,s in enumerate(nonmember_sample) if len(ngrams(s).intersection(sol_chars)) > 0]
            TPR_tr = 100 * len(m) / len(member_sample)
            FPR_tr = 100 * len(nm) / len(nonmember_sample)
            TPR = 100 * len([s for s in member_test if len(ngrams(s).intersection(sol_chars)) > 0]) / len(member_test)
            FPR = 100 * len([s for s in nonmember_test if len(ngrams(s).intersection(sol_chars)) > 0]) / len(nonmember_test)


            print(f"train TPR = {TPR_tr:.2f} @ {FPR_tr:.2f} FPR", f"test TPR = {TPR:.2f} @ {FPR:.2f} FPR")
            
            if FPR > TARGET_FPR:
                RES.append(best_tpr)
                y_true = [1]*len(member_test) + [0]*len(nonmember_test)
                y_score = [1 if len(ngrams(s).intersection(sol_chars)) > 0 else 0 for s in member_test] + \
                          [1 if len(ngrams(s).intersection(sol_chars)) > 0 else 0 for s in nonmember_test]
                auc_val = roc_auc_score(y_true, y_score)
                AUCS.append(auc_val)
                print(f"Run AUC: {auc_val:.4f}")  # NEW
                print("BEST TPR:", best_tpr)
                break
            else:
                best_tpr = max(best_tpr, TPR)

    print(RES)
    if len(RES) == 0:
        print(f'TPR@{fpr_budget}%FPR over {CROSS_VALS} runs: 0.0')
    else:
        print(f'TPR@{fpr_budget}%FPR over {CROSS_VALS} runs: {np.mean(RES):.4f} ± {np.std(RES):.4f}')

    if len(AUCS) > 0:  # NEW
        print(f"All AUCs: {AUCS}")
        print(f"Mean AUC over {CROSS_VALS} runs: {np.mean(AUCS):.4f} ± {np.std(AUCS):.4f}")  # NEW

    return np.mean(AUCS), np.std(AUCS), np.mean(RES), np.std(RES)