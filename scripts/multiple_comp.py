#plot histograms and box plots of features from last generation of different experiments

#todo:
#check with 3 experiments
#verify statistical significance test
#are the statistics OK? (final populations of x runs combined and compared with other method? )

import os
import sys
import csv
from shutil import copyfile
from string import Template

import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
from scipy import stats
import statistics
import math

import statsmodels.api as sm
from statsmodels.formula.api import ols

from statsmodels.stats.multicomp import (pairwise_tukeyhsd,
                                         MultiComparison)


import pandas as pd
import glob

#from jbmouret tutorial
def stars(p):
   if p < 0.0001:
       return "****"
   elif (p < 0.001):
       return "***"
   elif (p < 0.01):
       return "**"
   elif (p < 0.05):
       return "*"
   else:
       return "-"

############################
# read from files
############################
def get_immediate_subdirectories(a_dir):

    return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]

def get_data_list(exp_name):
    data_lists = {}
    for folder in get_immediate_subdirectories(exp_name):
        data_lists[folder]=[]
        csv_files = ["generations.csv","robots.csv"] #the rest is hardcoded for these two files for now
        for datafile in csv_files:
            fn = os.path.join(exp_name,folder,datafile)
            data_lists[folder].append(pd.read_csv(fn, index_col=False))

    return data_lists


def get_last_generation(data_lists, fitness_function):
    last_gens = {}
    first_gens = {}

    for exp in sorted(data_lists.keys()):
        merged = pd.merge(data_lists[exp][0], data_lists[exp][1], left_on=['robot_id'], right_on=['id'])


        merged["fitness_function"] = fitness_function
        merged["world"] = exp[0:6]
        max_gen = merged['gen'].idxmax()
        last_gen_idx=merged['gen'][max_gen]
        last_gens[exp]=merged[merged['gen']==last_gen_idx]
        first_gens[exp]=merged[merged['gen']==0]
    return last_gens


frames = []
for fitness_function in sys.argv[1:]:
    data_lists = get_data_list(fitness_function)
    last_gens = get_last_generation(data_lists, fitness_function)
    fitness_func_frame = pd.concat(last_gens)
    if 'bbox_max_x' in fitness_func_frame.columns:
         fitness_func_frame = fitness_func_frame.drop(columns=['bbox_max_z', 'bbox_min_z', 'bbox_max_x', 'bbox_max_y', 'bbox_min_x', 'bbox_min_y'])
    print("TYPE", type(fitness_func_frame))
    frames.append(fitness_func_frame)


#print(frames[0])
df = pd.concat(frames, ignore_index=True)

comb = pd.concat(last_gens)
measures=['vel','dvel','joint_count','extremity_count','nparts']

for measure in measures:
    # Set up the data for comparison (creates a specialised object)
    print(measure)
    MultiComp = MultiComparison(df[measure],
                                df['fitness_function'])

    # Show all pair-wise comparisons:

    # Print the comparisons

    print(MultiComp.tukeyhsd().summary())
    #mod = ols((measure + "~ fitness_function"), data=df).fit()
    #aov_table = sm.stats.anova_lm(mod, typ=2)
    #print("MEASURE:", measure)
    #print(aov_table)
