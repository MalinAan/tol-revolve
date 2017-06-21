#merge results from runs on different computers to a single directory
#-merge the .csv files (have only one header, sorted runs)
#-copy all robot files into same directory
#currently not copying in order of the runs, so csv files are not sorted on runs

import os
import sys
import csv
from shutil import copyfile
from string import Template

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]

#file structure:
#-src_folder
#  -worker1
#    -exp_name
#  -...
#  -workerN
#    -exp_name(same for all workers)

if len(sys.argv) != 3:
    print "usage: merge_runs dst_folder src_folder  (where src_folder contains a folder for each )"
    sys.exit(1)

#src: path to multiple folders
#dst: path to merged folder
#src_out_dir = "exp_par"
#dst_dir = "exp_merged"
dst_dir = sys.argv[1]
src_out_dir = sys.argv[2]
print dst_dir
print src_out_dir
worker_dirs = get_immediate_subdirectories(src_out_dir)

#find experiment name, which is a subdirectory of the worker directories
exp_name=get_immediate_subdirectories(os.path.join(src_out_dir,worker_dirs[0]))[0]

#create destination merged directory
dst_path = os.path.join(dst_dir,exp_name)
print "destination directory:", dst_path
if not os.path.exists(dst_path):
    os.makedirs(dst_path)
else:
    print "Destination directory already exists! Don't do it?"
    sys.exit(1)

#process all worker directories
for worker in worker_dirs:
    print "processing worker: ", worker

    #find csvs
    src_path = os.path.join(os.path.join(src_out_dir,worker,exp_name))
    print "source directory: ", src_path
    csv_files = [s for s in os.listdir(src_path) if '.csv' in s]
    print "copying and merging csv files: ", csv_files
    for f in csv_files:
        src_file = os.path.join(src_path,f)
        #print "source: ", src_file
        dst_file = os.path.join(dst_path,f)
        #print "destination: ", dst_file

        if not os.path.isfile(dst_file):
            print "destination file does not exist. copying.."
            copyfile(src_file,dst_file)
        else:
            print "destination file exists. appending."
            with open(dst_file,"a") as dst_f:
                with open(src_file,"r") as src_f:
                    src_f.next() #skip header line
                    for line in src_f:
                            dst_f.write(line);

    #copy .conf, don't worry about overwrite
    copy_files = [s for s in os.listdir(src_path) if ('.conf' in s or '.world' in s)]
    for f in copy_files:
        print "copying ", f
        copyfile(os.path.join(src_path,f),os.path.join(dst_path,f))

    #copy .sdf and .pb robot files, check if similar filenames from different runs
    print "copying robot files..."
    robot_count = 0
    robot_duplicates = 0
    copy_files = [s for s in os.listdir(src_path) if ('.pb' in s or '.sdf' in s)]
    for f in copy_files:
        #print "copying ", f
        dst = os.path.join(dst_path,f)
        if not os.path.isfile(dst):
            copyfile(os.path.join(src_path,f),dst)
            robot_count += 1
        else:
            robot_duplicates += 1
    print "copied %d robots." % (robot_count/2)
    if robot_duplicates > 0:
        print "WARNING! there were %d duplicate robot IDs which were not copied." % robot_duplicates
