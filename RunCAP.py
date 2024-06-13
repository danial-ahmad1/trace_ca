import CAProcessor as cap
import tkinter as tk
from tkinter import filedialog

def select_file():
    root = tk.Tk()
    root.withdraw() 
    file_path = filedialog.askopenfilename(
        filetypes=[("IMS files", "*.ims")]  # Only allow .tif files
    )  
    return file_path

selected_path = select_file()
CA_processor = cap.CAProcessor(selected_path)
CA_processor.run_CAProcessor()

# If in Jupyter Notebook, use the following to clear memory
# import gc
# del img_processor
# gc.collect()