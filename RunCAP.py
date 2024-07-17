import CAProcessor as cap
import tkinter as tk
from tkinter import filedialog
import os
import pandas as pd

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

folder_loc = select_folder()
files_list = pull_files(folder_loc)

print('Folder Location: ', folder_loc)
print('Files List: ', files_list)

files_expname = []
files_expname = [f.split('_')[0] for f in files_list]
files_expname_unique = list(set(files_expname))

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
    meta_analysis[CA_processor.file_name_trunc].append(len(CA_processor.region_im_filtered))

    if CA_processor.area_sum > 0:
        meta_analysis2[CA_processor.file_name_trunc].append(CA_processor.area_mean)
    elif CA_processor.area_sum == 0:
        meta_analysis2[CA_processor.file_name_trunc].append(0)

    # Particles per window size (1400x1400 pix = 238.1x238.1 µm), mm^2.
    meta_analysis3[CA_processor.file_name_trunc].append(len(CA_processor.region_im_filtered)/(.2381*.2381))

    # meta_analysis3[CA_processor.file_name_trunc].append(flatten(CA_processor.area_list))

    # CA_processor.run_CAPDebug_fcn1()
    # CA_processor.run_CAPDebug_fcn2()
    # CA_processor.run_CAPDebug_fcn3()
    # meta_analysis.append(CA_processor.file_name_trunc + ', Edge Mean: ' + str(CA_processor.edges_sum)) # With debug function 3

meta_df = pd.DataFrame(meta_analysis)
meta_df.to_csv('/Users/moose/Desktop/trace_ca-local/Computed_Results' + '/meta_analysis.csv', index=False)
print(meta_df)

meta2_df = pd.DataFrame(meta_analysis2)
meta2_df.to_csv('/Users/moose/Desktop/trace_ca-local/Computed_Results' + '/meta_analysis2.csv', index=False)
print(meta2_df)

meta3_df = pd.DataFrame(meta_analysis3)
meta3_df.to_csv('/Users/moose/Desktop/trace_ca-local/Computed_Results' + '/meta_analysis3.csv', index=False)
print(meta3_df)

# print(meta_analysis)
# If in Jupyter Notebook, use the following to clear memory
# import gc
# del img_processor
# gc.collect()