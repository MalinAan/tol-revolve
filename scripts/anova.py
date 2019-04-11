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
for folder in get_immediate_subdirectories(sys.argv[1]):
    #print folder
    data_lists[folder]=[]
    csv_files = ["generations.csv","robots.csv"] #the rest is hardcoded for these two files for now
    for datafile in csv_files:
        fn = os.path.join(sys.argv[1],folder,datafile)
        # print fn
        data_lists[folder].append(pd.read_csv(fn, index_col=False))
#print list(data_lists)
        

##############################################
#merge and extract first and last generations
##############################################
last_gens = {}
first_gens = {}
print("keys", data_lists.keys())
print("0", data_lists["world0_old_fitness"][0])
print("0", data_lists["world0_old_fitness"][1])
for exp in data_lists.keys():
    #print exp
    #merging the two data frames. Assuming only two now
    #not perfect, should not duplicate the run columns
    merged = pd.merge(data_lists[exp][0], data_lists[exp][1], left_on=['robot_id'], right_on=['id'])
    
    merged["world"] = exp
    if(exp == "world0_old_fitness"):
        print ("LIST", list(merged))
        print("MERGED", merged)
    #find and extract last and first generations
    max_gen = merged['gen'].idxmax()
    last_gen_idx=merged['gen'][max_gen]
    #print "last gen:", last_gen_idx
    last_gens[exp]=merged[merged['gen']==last_gen_idx]
    first_gens[exp]=merged[merged['gen']==0]

    #checking the data
    #print "run count: ", merged['run_x'].nunique()
    #print "number of first generations extracted: ", last_gens[exp]['run_x'].nunique()
    #print "number of last generations extracted: ", last_gens[exp]['run_x'].nunique()
    #print "number of solutions over all runs: ",  len(last_gens[exp])

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


#now try to merge all exps
#print(last_gens)
#print(last_gens.to_string())
comb = pd.concat(last_gens)

#print(comb["robot_id"])
#print(comb.to_string())
#print(comb["run_x"])
#print("UNNAMED", comb["unnamed"])
#print(vars(comb))



data = sm.datasets.get_rdataset("dietox", "geepack").data

#print(data)
md = smf.mixedlm("Weight ~ Time", data, groups=data["Pig"])

mdf = md.fit()

print(mdf.summary())

md_robots = smf.mixedlm("fitness ~ vel+nparts+world", comb, groups=comb["robot_id"])

mdf_robots = md_robots.fit()

print(mdf_robots.summary())