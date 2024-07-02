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

folder_loc = select_folder()
files_list = pull_files(folder_loc)

print('Folder Location: ', folder_loc)
print('Files List: ', files_list)

files_expname = []
files_expname = [f.split('_')[0] for f in files_list]
files_expname_unique = list(set(files_expname))

meta_analysis = {}
for expname in files_expname_unique:
    meta_analysis[expname] = []

for file in files_list:
    selected_path = folder_loc + '/' + file
    CA_processor = cap.CAProcessor(selected_path)
    CA_processor.run_CAProcessor()
    meta_analysis[CA_processor.file_name_trunc].append(len(CA_processor.region_im_filtered))

    # CA_processor.run_CAPDebug_fcn1()
    # CA_processor.run_CAPDebug_fcn2()
    # CA_processor.run_CAPDebug_fcn3()
    # meta_analysis.append(CA_processor.file_name_trunc + ', Edge Mean: ' + str(CA_processor.edges_sum)) # With debug function 3

meta_df = pd.DataFrame(meta_analysis)
meta_df.to_csv('/Users/moose/Desktop/trace_ca-local/Computed_Results' + '/meta_analysis.csv', index=False)
print(meta_df)

# print(meta_analysis)
# If in Jupyter Notebook, use the following to clear memory
# import gc
# del img_processor
# gc.collect()