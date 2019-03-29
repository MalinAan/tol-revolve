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
#measures=['fitness','vel','dvel','joint_count','extremity_count','nparts','t_eval']
#measures=['joint_count','nparts']



f, axarr = plt.subplots(len(measures),2,sharey='row')
nbins=30
coord=0
for m in measures: #one plot for each measure
    firstgens = []
    lastgens = []
    combined = []

    keys=sorted(last_gens.keys())
    combined_labels=[]

    for exp in keys: #combine plots for each experiment
        firstgens.append(first_gens[exp][m])
        lastgens.append(last_gens[exp][m])
        combined.append(first_gens[exp][m])
        combined.append(last_gens[exp][m])
        combined_labels.append(exp+"-first")
        combined_labels.append(exp+"-last")

    print combined_labels        
    
    #test violin plot
    axarr[coord,0].violinplot(firstgens)
    axarr[coord,1].violinplot(lastgens)
    set_axis_style(axarr[coord,0],sorted(first_gens.keys()),"first gen")
    set_axis_style(axarr[coord,1],sorted(last_gens.keys()),"last gen")

#    axarr[coord].violinplot(combined)
#    set_axis_style(axarr[coord],combined_labels)


    #try to get first and last gen 

 

    coord +=1

#set titles to the left
for ax, row in zip(axarr[:,0], measures):
    ax.set_ylabel(row, rotation='vertical', size='medium')

f.subplots_adjust(hspace=0.8)
#plt.show()
f.set_size_inches(16,28) #arbitrary paper size
plt.savefig("violin_test.pdf")

#%%
