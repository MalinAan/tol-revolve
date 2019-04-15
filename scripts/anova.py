import statsmodels.api as sm

import statsmodels.formula.api as smf

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



import pandas as pd
import glob



def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]


def print_stats(data, exp, measure):
    print("Experiment: %s", exp)
    print("Measure %s", measure)
    data = np.array(data)
    desc = stats.describe(data)
    print('# of observations:', desc.nobs)
    print('min: %d\nmax: %d' % desc.minmax)
    print('mean: %.1f' % desc.mean)
    print('variance: %.1f' % desc.variance)
    print('stdev: %.1f' % math.sqrt(desc.variance))
    print('median: %.1f' % statistics.median(data))



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
    #print("keys", data_lists.keys())
    #print("0", data_lists["world0_old_fitness"][0])
    #print("0", data_lists["world0_old_fitness"][1])
    for exp in data_lists.keys():
        merged = pd.merge(data_lists[exp][0], data_lists[exp][1], left_on=['robot_id'], right_on=['id'])
        

        merged["fitness_function"] = fitness_function
        merged["world"] = exp[0:6]
        #if(exp == "world0_old_fitness"):
            #print ("LIST", list(merged))
            #print("MERGED", merged)
        #find and extract last and first generations
        max_gen = merged['gen'].idxmax()
        last_gen_idx=merged['gen'][max_gen]
        #print "last gen:", last_gen_idx
        last_gens[exp]=merged[merged['gen']==last_gen_idx]
        first_gens[exp]=merged[merged['gen']==0]
    return last_gens


fitness_func_folders = sys.argv[1:]
print(fitness_func_folders)

frames = []
for fitness_func in fitness_func_folders:
    data_lists = get_data_list(fitness_func)
    last_gens = get_last_generation(data_lists, fitness_func)
    fitness_func_frame = pd.concat(last_gens)
    print("FITNESS_FUNC", fitness_func)
    fitness_func_frame = fitness_func_frame.drop(columns=['run_y', 'id'])
    
    fitness_func_desc = fitness_func_frame.drop(columns=['outputs', 'robot_id', 'conn', 'parent1', 'inputs', 'motor_count', 'parent2', 'gen', 'hidden'])
    if 'bbox_max_x' in fitness_func_frame.columns:
         fitness_func_desc = fitness_func_desc.drop(columns=['bbox_max_z', 'bbox_min_z', 'bbox_max_x', 'bbox_max_y', 'bbox_min_x', 'bbox_min_y'])
    print(fitness_func_desc.describe().to_string())
    md_robots = smf.mixedlm("fitness ~ vel+nparts+world", fitness_func_frame, groups=fitness_func_frame["robot_id"])

    mdf_robots = md_robots.fit()

    print(mdf_robots.summary())

    frames.append(fitness_func_frame)    



#now try to merge all exps
#print(last_gens)
#print(last_gens.to_string())
comb = pd.concat(frames)
#print(comb.describe().to_string())
#comb = comb.drop(columns=['run_y', 'id'])
print(comb.describe().to_string())

#print(comb["robot_id"])
#print(comb.to_string())
#print(comb["world"])
#print("UNNAMED", comb["unnamed"])
#print(vars(comb))



#data = sm.datasets.get_rdataset("dietox", "geepack").data

#print(data)
#md = smf.mixedlm("Weight ~ Time", data, groups=data["Pig"])
#mdf = md.fit()
#print(mdf.summary())

md_robots = smf.mixedlm("vel ~ fitness_function+world", comb, groups=comb["world"])

mdf_robots = md_robots.fit()

print(mdf_robots.summary())