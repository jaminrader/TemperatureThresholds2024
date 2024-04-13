import os
import base_directories

dir_settings = base_directories.get_directories()

for key in dir_settings:
    if key != "data":
        mkdir_folder = dir_settings[key]
        os.system('mkdir ' + mkdir_folder)