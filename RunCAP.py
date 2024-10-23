# Header
# Author: Dan Ahmad, PhD - For the University of Rochester (UR) - BME Department - TRaCE-bmps
# Version 1.1, Aug 7th 2024
# Runs on Python 3.11.9

# Edited to remove unecessary modules 7/17/24
import CAProcessor as cap
import tkinter as tk
from tkinter import filedialog
import os
import csv
from scipy.stats import f_oneway, ttest_ind
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

folder_loc = []
files_list = []

# Currently uses tkinter, but will be updated for either a CLI or watcher script in the future.
# So this function can be dropped in later versions.
def select_folder():
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory()

    return folder_path

def pull_files(folder_path):
    # Get all ims files in the folder
    files = os.listdir(folder_path)
    files_im = [f for f in files if f.endswith('.ims')]
    
    meta_list = []
    # Get all metadata files in the folder and associate them with an ims file
    for i in range(len(files_im)):
        meta_name = files_im[i][:-4] + '_metadata.txt'
        
        if meta_name in files:
            meta_list.append(meta_name)
        else:
            print('Warning! No metadata file found for: ', files_im[i])

    group_metadata = [f for f in files if f.endswith('.csv')]
    if group_metadata:
        print('Group metadata file found.')
    else:
        print('Warning! No group metadata file found.')
        group_metadata = None
    
    return files_im, meta_list, group_metadata


# Legacy code, not used in this version of the program, but keep it just in case.
# Flatten a list of lists
def flatten(lst):
    flat_list = []
    for item in lst:
        if isinstance(item, list):
            flat_list.extend(flatten(item))
        else:
            flat_list.append(item)
    return flat_list
#-----------------------------------------------------------------------------------

def name_sorter(files_list_input):
    wtlist = ['WT', 'wt', 'Wt', 'wT', 'wild type', 'Wild Type', 'Wild-type', 'wild-Type', 'WildType', 'Wild-type', 'wild_type']
    pbp4list = ['PBP4', 'pbp4', 'Pbp4', 'pBp4', 'PBP 4', 'pbp 4', 'Pbp 4', 'pBp-4', 'PBP-4', 'pbp-4']
    nplist = ['NP', 'np', 'nonporous', 'Nonporous', 'NonPorous', 'nonPorous', 'Non-Porous', 'non-porous', 'Non-porous', 'Non_Porous', 'non_Porous']
    dnaselist = ['DNAse', 'dnase', 'DNASE', 'DNASe', 'DNase', 'Dnase', 'dNaSe', 'DnAsE']
    experiment_group = []
    found_wt = False
    found_pbp4 = False
    found_np = False
    found_dnase = False

    for name in files_list_input:
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

    if not found_wt and not found_pbp4 and not found_np and not found_dnase:
        experiment_group = None
        
    return experiment_group

folder_loc = select_folder()
files_list, meta_info, group_meta = pull_files(folder_loc)

print('Folder Location: ', folder_loc)
if group_meta == None:
    print('No device metadata found. Check your file upload!')
else:
    print('Group Metadata: ', group_meta)
print('Files List: ', files_list)
print('Metadata List: ', meta_info)

assert group_meta != None, 'Warning! No device metadata found. Check your file upload! Quitting program...'

files_expname_unique = name_sorter(files_list)
if files_expname_unique == None:
    print('No experiment types found in file names, searching device metadata')
    
else:
    print('Experiment Names:  ', files_expname_unique)

files_list.sort()
meta_info.sort()
df_init = pd.read_csv(folder_loc + '/' + group_meta[0])
df1 = df_init.sort_values(df_init.columns[0], ascending = True)
df1 = df1.drop(columns=['Description', 'Membrane Type', 'Membrane Lot'])

print(df1)

for col in df1.columns[1:]:  # Exclude the first column (experiment ID number)
    if len(df1[col].unique()) == 1:
        print(f"Column '{col}' has unique value {df1[col].unique()[0]}")

# Filter out constant columns
print('Filtering out unchanged experimental parameters...')
constant_cols = [col for col in df1.columns[1:] if len(df1[col].unique()) == 1]
df_fullmeta = df_init.sort_values(df_init.columns[0], ascending = True)
df1 = df1.drop(constant_cols, axis=1)
print('Using the following groups for data separation...')
for col in df1.columns[1:]:
    print(f"{col}")

grab_expnames = []
for index, row in df1.iloc[:, 1:].iterrows():
    grab_expnames.append(tuple(row))

if files_expname_unique == None:
    files_expname_unique = []
    files_expname_group = list(set(grab_expnames))

def flatten_df_string(list_of_strings):
    return ', '.join(list_of_strings)

for i in range(len(files_expname_group)):
    files_expname_unique.append(flatten_df_string(files_expname_group[i]))

print(f"Experiment names are {files_expname_unique}")

step_size = {}
if len(meta_info) > 0:
    for i in range(len(files_list)):
        # Check if there's a metadata file for this experiment
        if any(files_list[i][:-4] in s for s in meta_info):
            # Open the metadata file
            with open(os.path.join(folder_loc, files_list[i][:-4] + '_metadata.txt')) as f:
                for line in f:
                    # Check for a StepSize line
                    if 'StepSize' in line:
                        print('File:',files_list[i],'has',line.strip())
                        # Add the step size string to the dictionary step_size, will need to clean up and turn to integer
                        step_size[files_list[i]] = line.strip()
                        break
        else:
            print('Warning: No metadata file found for:', files_list[i], ', step size is set to 0.2 µm.')
            # If no metadata file was found, add a default value to the dictionary
            step_size[files_list[i]] = 'StepSize=0.2'
else:
    print('Warning: No metadata files found in folder. Step size will be set to 0.2 µm for all files...')
    # Add the default step size to the dictionary if no metadata files were found for n experiments (size of files_list array)
    for file in files_list:
        step_size[file] = 'StepSize=0.2'

# Convert text to number
for i in range(len(files_list)):    
    step_size[files_list[i]] = float(step_size[files_list[i]].split('StepSize=')[1])

meta_analysis = {expname: [] for expname in files_expname_unique} # Number of cluster detections per image
meta_analysis2 = {expname: [] for expname in files_expname_unique} # Number of detections per area
meta_analysis3 = {expname: [] for expname in files_expname_unique} # All detection sizes per image

device_list = []
for i in range(len(files_list)):
    device_list.append([])

for i in range(len(files_list)):
    device_list[i].append(files_list[i].split('_')[0])

iter = 0
for index, row in df1.iloc[:, 1:].iterrows():
    device_list[iter].append(tuple(row))
    iter += 1

for i in range(len(device_list)):
    device_list[i][1] = flatten_df_string(device_list[i][1])

particles_per_window = []
for file in files_list:
    selected_path = folder_loc + '/' + file
    CA_processor = cap.CAProcessor(selected_path, step_size[file])
    CA_processor.run_CAProcessor()

    if len(CA_processor.region_im_filtered) > 0:
        for i in range(len(device_list)):
            if device_list[i][0] == CA_processor.file_name_trunc:
                meta_analysis[device_list[i][1]].append(len(CA_processor.region_im_filtered))
                break

    elif len(CA_processor.region_im_filtered) == 0:
        for i in range(len(device_list)):
            if device_list[i][0] == CA_processor.file_name_trunc:
                meta_analysis[device_list[i][1]].append(0)
                break

    # Mean particle size in µm^2
    if CA_processor.area_sum > 0:
        for i in range(len(device_list)):
            if device_list[i][0] == CA_processor.file_name_trunc:
                meta_analysis2[device_list[i][1]].append(CA_processor.area_mean * ((238.1/1400)**2))
                break

    elif CA_processor.area_sum == 0:
        for i in range(len(device_list)):
            if device_list[i][0] == CA_processor.file_name_trunc:
                meta_analysis2[device_list[i][1]].append(0)
                break

    # Particles per window size (1400x1400 pix = 238.1x238.1 µm), mm^2.
    if len(CA_processor.region_im_filtered) > 0:
        particles_per_window.append(len(CA_processor.region_im_filtered)/(.2381*.2381))
        for i in range(len(device_list)):
            if device_list[i][0] == CA_processor.file_name_trunc:
                meta_analysis3[device_list[i][1]].append(len(CA_processor.region_im_filtered)/(.2381*.2381))
                break

    elif len(CA_processor.region_im_filtered) == 0:
        particles_per_window.append(0)
        for i in range(len(device_list)):
            if device_list[i][0] == CA_processor.file_name_trunc:
                meta_analysis3[device_list[i][1]].append(0)
                break

print("Stack processing complete, creating meta analyses...")

# Generate csv files for each test
meta_list = [meta_analysis, meta_analysis2, meta_analysis3]
save_name = ['Number_of_Detections', 'Mean_Particle_Size_um2', 'Particles_per_mm2']
save_name_index = 0
df_fullmeta['Particles/mm^2'] = particles_per_window
df_fullmeta.to_csv(CA_processor.savepath + 'device_metadata_results.csv', index = False)

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
    with open(CA_processor.savepath + save_name[save_name_index] + '.csv', 'w', newline='') as csvfile:
        fieldnames = list(test.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    save_name_index += 1

# Create graph with all data points and ANOVA for particles/mm^2 detected

anova_groups = [meta_analysis2[key] for key in meta_analysis2.keys()]

# Only perform ANOVA if number of experimental groups is greater than three, otherwise do a t-test for n=2, and nothing for n=1
# We'll also make a graph for the ANOVA test
if len(anova_groups) > 2:
    anova_result = f_oneway(*anova_groups)
    print(f"F-statistic: {anova_result.statistic}, P-value: {anova_result.pvalue}")

    # Tukey HSD
    tukey_values = np.concatenate(anova_groups) 
    tukey_labels = np.concatenate([[key] * len(sorted(meta_analysis3[key])) for key in sorted(meta_analysis3.keys())])  
    tukey_result = pairwise_tukeyhsd(tukey_values, tukey_labels)

    groups = [key for key in sorted(meta_analysis3.keys())]
    maxes = [np.max(sorted(meta_analysis3[key])) for key in sorted(meta_analysis3.keys())]

    # Convert the tukey hsd results to a df to help organize
    tukey_df = pd.DataFrame(data=tukey_result.summary().data[1:], columns=tukey_result.summary().data[0])
    # Filter out significant comparisons only, showing NS probably not needed?
    significant_comparisons = tukey_df[tukey_df['reject'] == True]

    colors = ['Black', 'Orange', 'Cyan', 'Salmon']
    darker_colors = ['Dark' + color if color != 'Black' else 'Black' for color in colors]
    fig, ax = plt.subplots(dpi=300)
    for i, key in enumerate(sorted(meta_analysis3.keys())):
        ax.bar(key, np.mean(sorted(meta_analysis3[key])), yerr = np.std(sorted(meta_analysis3[key])), capsize = 10, alpha = 0.75, color=colors[i], width = 0.65, edgecolor = 'Black')
        for j in range(len(sorted(meta_analysis3[key]))):
            ax.scatter(key, sorted(meta_analysis3[key])[j], color = darker_colors[i], edgecolor = 'Black', s = 15)

    y_offset = 0 
    # Calculate the distance between groups for each comparison, used to sort signficance bars from closest group to farthest
    significant_comparisons['distance'] = significant_comparisons.apply(lambda row: abs(groups.index(row['group1']) - groups.index(row['group2'])) if not pd.isna(abs(groups.index(row['group1']) - groups.index(row['group2']))) else float('nan'), axis=1)

    for index, row in significant_comparisons.iterrows():
        group1, group2 = row['group1'], row['group2']
        x1, x2 = groups.index(group1), groups.index(group2)

        # For some reason the pairwise comparisons are randomly sorted
        # sorter = [(group1, x1), (group2, x2)] # If we don't sort, then the significance bars will be drawn in a random order
        # if sorter[0][1] > sorter[1][1]:
        #     sorter = sorter[::-1]
        #     group1, x1 = sorter[0]
        #     group2, x2 = sorter[1]

        max1, max2 = maxes[x1], maxes[x2]
        base_y = max(max1, max2) + 40  # Base y-position for the significance line
        y = max(base_y, y_offset)  # Adjust y-position based on offset to avoid overlap
        h, col = 15, 'Black'  # Height and color of the significance marker
        
        # Draw horizontal line for significance
        plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
        # Add significance marker "*", haven't figured out how to change based on p value, probably a list taken from the df?
        plt.text((x1+x2)*.5, y+h, "*", ha='center', va='bottom', color=col)
        
        # Draw vertical lines down to the bars
        # plt.plot([x1, x1], [y, max1+25], lw=1.5, c=col, linestyle='solid')  # Line down to group1 bar, not needed. Kept here just in case
        # plt.plot([x2, x2], [y, max2+25], lw=1.5, c=col, linestyle='solid')  # Line down to group2 bar, not needed. Kept here just in case
        
        y_offset = y + h + 100  # Increment y_offset for the next line so significance bars don't overlap

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)
    plt.title('Cluster Detections')
    plt.xlabel('Group')
    plt.ylabel('Clusters/mm^2')
    plt.savefig(os.path.join(CA_processor.savepath, 'Stats.png'))

elif len(anova_groups) == 2:
    ttest_result = ttest_ind(*anova_groups)
    print(f"T-statistic: {ttest_result.statistic}, P-value: {ttest_result.pvalue}")

    groups = [key for key in meta_analysis3.keys()]
    maxes = [np.max(meta_analysis3[key]) for key in meta_analysis3.keys()]
    vals = [meta_analysis3[key] for key in meta_analysis2.keys()]

    if ttest_result.pvalue < 0.05:
        print("There is a significant difference between the groups")
        colors = ['Black', 'Orange', 'Cyan', 'Salmon']
        darker_colors = ['Dark' + color if color != 'Black' else 'Black' for color in colors]
        fig, ax = plt.subplots(dpi=300)
        for i, key in enumerate(meta_analysis3.keys()):
            ax.bar(key, np.mean(meta_analysis3[key]), yerr = np.std(meta_analysis3[key]), capsize = 10, alpha = 0.75, color=colors[i], width = 0.65, edgecolor = 'Black')
            for j in range(len(meta_analysis3[key])):
                ax.scatter(key, meta_analysis3[key][j], color = darker_colors[i], edgecolor = 'Black', s = 15)

        y_offset = 0 

        x1 = 0
        x2 = 1

        max1, max2 = maxes[0], maxes[1]
        base_y = max(np.max(vals[0]), np.max(vals[1])) + 100  # Base y-position for the significance line
        y = base_y  # Adjust y-position based on offset to avoid overlap
        h, col = 15, 'Black'  # Height and color of the significance marker

        plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
        plt.text((x1+x2)*.5, y+h, "*, p < 0.05", ha='center', va='bottom', color=col)

        # Draw vertical lines down to the bars
        plt.plot([x1, x1], [y, np.max(vals[0])+50], lw=1.5, c=col, linestyle='solid')  # Line down to group1 bar
        plt.plot([x2, x2], [y, np.max(vals[1])+50], lw=1.5, c=col, linestyle='solid')  # Line down to group2 bar

        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(groups)
        plt.title('Cluster Detections')
        plt.xlabel('Group')
        plt.ylabel('Clusters/mm^2')
        plt.savefig(os.path.join(CA_processor.savepath,'Stats.png'))
    else:
        print("There is no significant difference between the groups")
        colors = ['Black', 'Orange', 'Cyan', 'Salmon']
        darker_colors = ['Dark' + color if color != 'Black' else 'Black' for color in colors]
        fig, ax = plt.subplots(dpi=300)
        for i, key in enumerate(meta_analysis3.keys()):
            ax.bar(key, np.mean(meta_analysis3[key]), yerr = np.std(meta_analysis3[key]), capsize = 10, alpha = 0.75, color=colors[i], width = 0.65, edgecolor = 'Black')
            for j in range(len(meta_analysis3[key])):
                ax.scatter(key, meta_analysis3[key][j], color = darker_colors[i], edgecolor = 'Black', s = 15)

        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(groups)
        plt.title('Cluster Detections')
        plt.xlabel('Group')
        plt.ylabel('Clusters/mm^2')
        plt.savefig(os.path.join(CA_processor.savepath,'Stats.png'))

elif len(anova_groups) == 1:
    print("Only one experimental group detected, cannot perform statistics testing")

else:
    print("No experimental groups detected, cannot perform statistics testing")

print("Meta analyses complete, csv files saved to designated folder.")
print("Graphs created with statistics.")
print("Please check the output folder for results.")
print("All processes complete, report generated, exiting program.")