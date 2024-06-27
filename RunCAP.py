import CAProcessor as cap
import tkinter as tk
from tkinter import filedialog
import os

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

for file in files_list:
    selected_path = folder_loc + '/' + file
    CA_processor = cap.CAProcessor(selected_path)
    CA_processor.run_CAProcessor()
    # CA_processor.run_CAPDebug_fcn1()

# If in Jupyter Notebook, use the following to clear memory
# import gc
# del img_processor
# gc.collect()