# Header
# Author: Dan Ahmad, PhD - For the University of Rochester (UR) - BME Department - TRaCE-bmps
# Version 1.0, June 21st 2024
# Runs on Python 3.11.8

# Edited to remove unecessary modules 7/17/24
import CAProcessor as cap
import tkinter as tk
from tkinter import filedialog
import os
import csv

folder_loc = []
files_list = []

def select_folder():
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory()

    return folder_path

def pull_files(folder_path):
    # Get all ims files in the folder
    files = os.listdir(folder_path)
    files = [f for f in files if f.endswith('.ims')]

    return files

def flatten(lst):
    flat_list = []
    for item in lst:
        if isinstance(item, list):
            flat_list.extend(flatten(item))
        else:
            flat_list.append(item)
    return flat_list

def name_sorter(files_list):
    wtlist = ['WT', 'wt', 'Wt', 'wT', 'wild type', 'Wild Type', 'Wild type', 'wild Type', 'wildtype', 'Wildtype', 'WildType', 'wild-type', 'Wild-type', 'Wild-type', 'wild-type', 'wild_Type', 'Wild_Type', 'Wild_Type', 'wild_Type']
    pbp4list = ['PBP4', 'pbp4', 'Pbp4', 'pBp4', 'PBP 4', 'pbp 4', 'Pbp 4', 'pBp 4', 'PBP-4', 'pbp-4', 'Pbp-4', 'pBp-4']
    nplist = ['NP', 'np', 'nonporous', 'Nonporous', 'NonPorous', 'nonPorous', 'Non-Porous', 'non-porous', 'Non-porous', 'Non_Porous', 'non_Porous', 'Non_Porous', 'non_Porous']
    dnaselist = ['DNAse', 'dnase', 'DNASE', 'DNASe', 'DNase', 'Dnase']
    experiment_group = []
    found_wt = False
    found_pbp4 = False
    found_np = False
    found_dnase = False

    for name in files_list:
        if any(x in name for x in wtlist) and not found_wt:
            experiment_group.append('Wild Type')
            found_wt = True
        elif any(x in name for x in pbp4list) and not found_pbp4:
            experiment_group.append('PBP4')
            found_pbp4 = True
        elif any(x in name for x in nplist) and not found_np:
            experiment_group.append('Nonporous')
            found_np = True
        elif any(x in name for x in dnaselist) and not found_dnase:
            experiment_group.append('DNAse')
            found_dnase = True

    return experiment_group

folder_loc = select_folder()
files_list = pull_files(folder_loc)

print('Folder Location: ', folder_loc)
print('Files List: ', files_list)

files_expname = []
files_expname = [f.split('_')[0] for f in files_list]
files_expname_all = list(set(files_expname))

files_expname_unique = name_sorter(files_expname_all)

# Number of detections
meta_analysis = {}
for expname in files_expname_unique:
    meta_analysis[expname] = []

# Number of detections per area
meta_analysis2 = {}
for expname in files_expname_unique:
    meta_analysis2[expname] = []

# All detection sizes per image
meta_analysis3 = {}
for expname in files_expname_unique:
    meta_analysis3[expname] = []

for file in files_list:
    selected_path = folder_loc + '/' + file
    CA_processor = cap.CAProcessor(selected_path)
    CA_processor.run_CAProcessor()

    if len(CA_processor.region_im_filtered) > 0:
        meta_analysis[CA_processor.file_name_trunc].append(len(CA_processor.region_im_filtered))
    elif len(CA_processor.region_im_filtered) == 0:
        meta_analysis[CA_processor.file_name_trunc].append(0)

    # Mean particle size in µm^2
    if CA_processor.area_sum > 0:
        meta_analysis2[CA_processor.file_name_trunc].append(CA_processor.area_mean * ((238.1/1400)**2))
    elif CA_processor.area_sum == 0:
        meta_analysis2[CA_processor.file_name_trunc].append(0)

    # Particles per window size (1400x1400 pix = 238.1x238.1 µm), mm^2.
    if len(CA_processor.region_im_filtered) > 0:
        meta_analysis3[CA_processor.file_name_trunc].append(len(CA_processor.region_im_filtered)/(.2381*.2381))
    elif len(CA_processor.region_im_filtered) == 0:
        meta_analysis3[CA_processor.file_name_trunc].append(0)

print("Stack processing complete, creating meta analyses...")

# Generate csv files for each test
meta_list = [meta_analysis, meta_analysis2, meta_analysis3]
save_name = ['Number_of_Detections', 'Mean_Particle_Size_um2', 'Particles_per_mm2']
save_name_index = 0

for test in meta_list:
    max_length = max(len(values) for values in test.values())

    # Make rows for csv using dictionary titles as headers, useful when pandas df can't handle uneven lists...
    rows = []
    for i in range(max_length):
        row = {}
        for key in test.keys():
            row[key] = test[key][i] if i < len(test[key]) else '' # Will fill in empty spaces with '' in csv in spots where there are no values if uneven lists are present
        rows.append(row)

    # Write to CSV
    with open('/Users/moose/Desktop/trace_ca-local/Computed_Results/' + save_name[save_name_index] + '.csv', 'w', newline='') as csvfile:
        fieldnames = list(test.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    save_name_index += 1

print("Meta analyses complete, csv files saved to designated folder.")
print("All processes complete, exiting program.")