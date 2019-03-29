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
import scipy.stats


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

data_lists = {}
for folder in get_immediate_subdirectories("output"):
    print folder
    data_lists[folder]=[]
    csv_files = ["generations.csv","robots.csv"] #the rest is hardcoded for these two files for now
    for datafile in csv_files:
        fn = os.path.join("output",folder,datafile)
        # print fn
        data_lists[folder].append(pd.read_csv(fn))
print list(data_lists)
        

##############################################
#merge and extract first and last generations
##############################################
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

#now try to merge all exps
comb = pd.concat(last_gens)

#important! how to select by indices and rows
# print comb.loc['world1',['fitness']]
# print comb.loc[:,['fitness']]

#need to read up on pandas to get the hang of things!

######################################
# plotting
######################################
plt.close('all')
measures=['fitness','vel','dvel','joint_count','extremity_count','nparts']
#measures=['fitness','vel','dvel','joint_count','extremity_count','nparts','t_eval']
#measures=['joint_count','nparts']


f, axarr = plt.subplots(len(measures),2)
nbins=30
coord=0
for m in measures: #one plot for each measure
    boxd = []
    m_range=[ float(comb.loc[:,[m]].min()), float(comb.loc[:,[m]].max()) ]
    print m
    print(m_range)
    #find a suitable number of bins for the measure
    if last_gens[last_gens.keys()[0]][m].dtype == np.float64 :
        nbins=30
    else:
        nbins = m_range[1] - m_range[0] + 1
        
    for exp in last_gens.keys(): #combine plots for each experiment
        #plot histogram of last gen
        plot_gens=last_gens[exp][m]
        boxd.append(plot_gens)
        axarr[coord,0].hist(plot_gens, int(nbins), m_range, alpha=0.5, label=exp, normed=1)
        #axarr[coord,0].set_title(m)
        if coord==0: axarr[coord,0].legend()

    #plot boxplot of last gen
    z, p = scipy.stats.mannwhitneyu(boxd[0], boxd[1], alternative="two-sided")
    print "mwu (z,p): ", z,p

    ax = axarr[coord,1]
    ax.boxplot(boxd, labels=last_gens.keys(), autorange=True)
    # axarr[coord,0].set_title(m)
    #if coord==0: axarr[coord,0].legend()

    y_max = np.max(np.concatenate((boxd[0],boxd[1])))
    y_min = np.min(np.concatenate((boxd[0],boxd[1])))
    ax.annotate("", xy=(1, y_max), xycoords='data',xytext=(2, y_max), textcoords='data',arrowprops=dict(arrowstyle="-", ec='#aaaaaa',connectionstyle="bar,fraction=0.1"))
    ax.text(1.5, y_max + abs(y_max - y_min)*0.2, stars(p),horizontalalignment='center',verticalalignment='center')

    coord +=1

#set titles to the left
for ax, row in zip(axarr[:,0], measures):
    ax.set_ylabel(row, rotation='vertical', size='medium')

f.subplots_adjust(hspace=0.8)
#plt.show()
f.set_size_inches(8,16) #arbitrary paper size
plt.savefig("measures_hist_box.pdf")





#good example of subplots:
#https://matplotlib.org/examples/pylab_examples/subplots_demo.html#pylab-examples-subplots-demo
# how to show in same plot?
