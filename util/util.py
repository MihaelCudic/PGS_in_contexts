import os
import torch

import numpy as np
import pandas as pd

from sklearn.metrics import r2_score
from scipy.stats import norm
import scipy.stats as stats
from scipy.stats import ttest_1samp

from bed_reader import open_bed

import subprocess

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def create_bins_and_indices(df, bin_definitions):
    all_labels = []
    label_to_index_map = {}
    cat_boundaries = []
    current_index = 0

    for category, bins in bin_definitions.items():
        if isinstance(bins, dict):
            cat_labels = [f"{category}: {label}" for label in bins.keys()]
            for label, bin in bins.items():
                if isinstance(bin, int):
                    label_to_index_map.update({f"{category}: {label}": df.index[df[category] == bin].tolist()})
                else:
                    label_to_index_map.update({f"{category}: {label}": df.index[df[category].isin(bin)].tolist()})
        else:
            cat_labels = [f"{category}: {bin[0]}-{bin[1]}" for bin in bins]
            label_to_index_map.update({f"{category}: {bin[0]}-{bin[1]}": df.index[(df[category] >= bin[0]) & (df[category] <= bin[1])].tolist() for bin in bins})
        all_labels.extend(cat_labels)
        cat_boundaries.append(current_index)
        current_index += len(cat_labels)
    cat_boundaries.append(current_index)

    return cat_boundaries, all_labels, label_to_index_map

def extract_1way_interactions(mat):
    if isinstance(mat, dict):
        vec_1way = {}
        for key, sub_dict in mat.items():
            vec_1way[key] = extract_1way_interactions(sub_dict)
    else:
        vec_1way = np.diagonal(mat, axis1=0, axis2=1)
        vec_1way = np.moveaxis(vec_1way, -1, 0)
    return vec_1way

def extract_2way_interactions(mat):
    if isinstance(mat, dict):
        vec_2way = {}
        for key, sub_dict in mat.items():
            vec_2way[key] = extract_2way_interactions(sub_dict)
    else:
        inds = np.tril_indices(mat.shape[0], k=-1)
        vec_2way_ = mat[inds[0], inds[1]]
        mask = ~np.isnan(vec_2way_.mean(axis=tuple(range(1, vec_2way_.ndim))))
        vec_2way = vec_2way_[mask]
    return vec_2way

def calc_empirical_pval(data1, data2=0, p_thresh=0.05, two_tail=True):
    diff_samples = data1 - data2
    num_samples = len(diff_samples)
    min_p_value = 1 / num_samples
    
    if two_tail:
        perct_ = np.mean(diff_samples < 0)
        p_value = 2 * min(perct_, 1 - perct_)
    else:
        # Assuming a one-sided test where we're only checking for values < 0
        p_value = np.mean(diff_samples < 0)
    
    p_value = max(p_value, min_p_value)
    return p_value < p_thresh, p_value

def compute_ci(data, confidence=0.95, axis=0):
    bot_percent = (1 - confidence) / 2 * 100
    top_percent = (1 + confidence) / 2 * 100
    ci = np.percentile(data, [bot_percent, top_percent], axis=axis)
    return ci[0], ci[1]