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

#%%###########################
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
        #print fn
        data_lists[folder].append(pd.read_csv(fn))
print list(data_lists)



#%%#########################################
# Print some stats of loaded experiments
############################################
for exp in data_lists.keys():
    print exp, ": ",
    df = data_lists[exp][0]
    dfr = data_lists[exp][1]

    print len(df), "entries, ",
    print df['gen'].max(), "generations, ",
    print df['run'].nunique(), "runs",
    print dfr['id'].nunique(), "robot infos"
#




#%%###########################################
#merge and extract first and last generations
##############################################
last_gens = {}
first_gens = {}
for exp in data_lists.keys():
    print exp
    #merging the two data frames. Assuming only two now
    #not perfect, should not duplicate the run columns
    cols_to_use = data_lists[exp][1].columns.difference(data_lists[exp][0].columns)
    merged = pd.merge(data_lists[exp][0], data_lists[exp][1][cols_to_use], left_on=['robot_id'], right_on=['id'])
    print list(merged)

    #find and extract last and first generations
    max_gen = merged['gen'].idxmax()
    last_gen_idx=merged['gen'][max_gen]
    print "last gen:", last_gen_idx
    first_gens[exp]=merged[merged['gen']==0]
    last_gens[exp]=merged[merged['gen']==last_gen_idx]

    #checking the data
    print "run count: ", merged['run'].nunique()
    print "number of first generations extracted: ", first_gens[exp]['run'].nunique()
    print "number of last generations extracted: ", last_gens[exp]['run'].nunique()
    print "number of solutions over all runs: ",  len(last_gens[exp])


#%%###################################
# plotting
######################################
#for setting labels to violin plots
def set_axis_style(ax, labels, title="sample name"):
    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel(title)


plt.close('all')
measures=['fitness','vel','dvel','joint_count','extremity_count','nparts']
#measures=['fitness','joint_count']

scenarios=sorted(last_gens.keys())
#scenarios=['world0','world1','world2']

f, axarr = plt.subplots(len(measures),len(scenarios), sharey='row')
coord=0
for m in measures: #one plot for each measure

    #loop scenarios
    coord_x=0
    for exp in scenarios: #combine plots for each experiment
        print(exp)     
    
        #try to plot all runs as separate violins
        #loop runs
        pops=last_gens[exp]
        lastgens = []
        for run in range(0,pops['run'].nunique()) : 
            lastgens.append(pops[pops['run']==run][m])    

#        axarr[coord,coord_x].violinplot(lastgens)
        axarr[coord,coord_x].boxplot(lastgens)
        axarr[coord,coord_x].set_xlabel(exp)
        coord_x +=1


    coord +=1




#set titles to the left
for ax, row in zip(axarr[:,0], measures):
    ax.set_ylabel(row, rotation='vertical', size='medium')

f.subplots_adjust(hspace=0.8)
#plt.show()
f.set_size_inches(40,28) #arbitrary paper size
plt.savefig("violin_populations.pdf")

#%%
