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
concated = pd.concat(frames, ignore_index=True)

comb = pd.concat(last_gens)


plt.close('all')
measures=['vel','dvel','joint_count','extremity_count','nparts']

exp_folder = "Final_exp"
exp_name = exp_folder[:-1]
path = os.getcwd()
new_dirname = exp_name + "_plots"
print("NEW DIR", (path + "/" + new_dirname))
os.mkdir((path + "/" + new_dirname))



f, axarr = plt.subplots(len(measures),2)
nbins=30
coord=0
for m in measures: #one plot for each measure


    m_range=[ float(comb.loc[:,[m]].min()), float(comb.loc[:,[m]].max()) ]
    print m
    print(m_range)
    #find a suitable number of bins for the measure
    #if last_gens[last_gens.keys()[0]][m].dtype == np.float64 :
    #    nbins=30
    #else:
    #    nbins = m_range[1] - m_range[0] + 1

    hist_fig = plt.figure(figsize=(14, 10))
    hist_fig.suptitle(('Histogram for ' + m) , fontsize=20, fontweight='bold')
    ax_hist = hist_fig.add_subplot(1,1,1)       #axarr[coord,0].set_title(m)fig.add_subplot(1,1,1)

    labels = []
    boxd = []
    for index, exp in enumerate(sys.argv[1:]): #combine plots for each experiment
        label = ("Fitness function " + str(index + 1) + " "+ exp)
        plot_gens=frames[index]
        boxd.append(plot_gens)
        labels.append(label)



    df = concated

    grouped = df.groupby('fitness_function')

    for index, group in enumerate(grouped):
        label = labels[index]
        ax_hist.hist(group[1][m], int(nbins), m_range, alpha=0.5, label=label, normed=1)
        if coord==0:
            axarr[coord,0].legend()


    ax_hist.legend(fontsize=18)
    #plt.show()

    for item in ([ax_hist.xaxis.label, ax_hist.yaxis.label] + ax_hist.get_xticklabels() + ax_hist.get_yticklabels()):
        item.set_fontsize(15)

    hist_fig.savefig((new_dirname + '/hist_'+m+ "_" + exp_name +".pdf"))

    #plot boxplot of last gen
    z, p = stats.mannwhitneyu(boxd[0], boxd[1], alternative="two-sided")
    print "mwu (z,p): ", z,p



    df = concated



    boxplot_fig = plt.figure(figsize=(14, 10))

    ax_boxplot = boxplot_fig.add_subplot(1,1,1)


    df.boxplot(column=m, ax=ax_boxplot, by="fitness_function", fontsize=15)

    boxplot_fig.suptitle(('Boxplot for ' + m) , fontsize=20, fontweight='bold')
    plt.title("")
    plt.xlabel("")
    plt.xticks(range(1, (len(labels) + 1)), labels)
    #plt.show()
    boxplot_fig.savefig((new_dirname + '/boxplot_'+m + "_" + exp_name +".pdf"))
