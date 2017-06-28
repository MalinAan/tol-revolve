#simple test of histogram
#and to plot generation time for single and dual core VMs

import os
import sys
import csv
from shutil import copyfile
from string import Template

import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np

import pandas as pd
import glob

# read data
data_lists = {}
for folder in glob.glob("output/*"):
    data_lists[folder.split("/")[1]] = []
    csv_files = ["generations.csv","robots.csv"] #the rest is hardcoded for these two files for now
    for datafile in csv_files:
        fn = os.path.join(folder,datafile)
        # print fn
        data_lists[folder.split("/")[1]].append(pd.read_csv(fn))
print list(data_lists)


#merge and extract first and last generations
last_gens = {}
first_gens = {}
for exp in data_lists.keys():
    print exp
    #merging the two data frames. Assuming only two now
    #not perfect, should not duplicate the run columns
    merged = pd.merge(data_lists[exp][0], data_lists[exp][1], left_on=['robot_id'], right_on=['id'])
    print list(merged)

    #find and extract last and first generations
    max_gen = merged['gen'].idxmax()
    last_gen_idx=merged['gen'][max_gen]
    print "last gen:", last_gen_idx
    last_gens[exp]=merged[merged['gen']==last_gen_idx]
    first_gens[exp]=merged[merged['gen']==0]

    #checking the data
    print "run count: ", merged['run_x'].nunique()
    print "number of first generations extracted: ", last_gens[exp]['run_x'].nunique()
    print "number of last generations extracted: ", last_gens[exp]['run_x'].nunique()
    print "number of solutions over all runs: ",  len(last_gens[exp])

#plot histograms, built-in pandas historam
# names=['fitness','vel','dvel','joint_count','extremity_count','nparts','t_eval']
# for exp in last_gens.keys():
#     plot_gens=last_gens[exp][names]
#     plot_gens.hist(alpha=0.7, bins=16)
#     plt.suptitle(exp)
# plt.show()
# exit(1)

#now try to merge all exps
comb = pd.concat(last_gens)

#important! how to select by indices and rows
# print comb.loc['world1',['fitness']]
# print comb.loc[:,['fitness']]

#need to read up on pandas to get the hang of things!

#start plotting
plt.close('all')
#measures=['fitness','vel','dvel','joint_count','extremity_count','nparts']
measures=['fitness','vel','dvel','joint_count','extremity_count','nparts','t_eval']
#measures=['joint_count','nparts']


f, axarr = plt.subplots(len(measures))
nbins=30
coord=0
for m in measures: #one plot for each measure
    for exp in last_gens.keys(): #combine plots for each experiment
        plot_gens=last_gens[exp][m]
        comb_range=[ float(comb.loc[:,[m]].min()), float(comb.loc[:,[m]].max()) ]
        axarr[coord].hist(plot_gens, nbins, comb_range, alpha=0.5, label=exp, normed=1)
        axarr[coord].set_title(m)
        if coord==0:
            axarr[coord].legend()
    coord +=1

f.subplots_adjust(hspace=1.0)
plt.show()





#good example of subplots:
# https://matplotlib.org/examples/pylab_examples/subplots_demo.html#pylab-examples-subplots-demo

# nbins=30
# for exp in last_gens.keys():
#     coord=0
#     for m in measures:
#         plot_gens=last_gens[exp][m]
#         axarr[coord].hist(plot_gens, nbins, alpha=0.5, label=exp)
#         if coord==0:
#             axarr[coord].legend()
#         coord += 1

# how to show in same plot?
# option 1: make dataframes per measure, containing all exps, then plot and subplot manually
# option 2: subplot and matplotlib manually
