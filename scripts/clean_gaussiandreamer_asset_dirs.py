import os
import re
import shutil

def clean_gaussian_dreamer_dirs(root_dir):
    # List of directories to keep
    keep_dirs = {'ckpts', 'save'}

    # Iterate over subdirectories in the root directory
    for sub_dir in os.listdir(root_dir):
        sub_dir_path = os.path.join(root_dir, sub_dir)
        
        # Check if the item is a directory
        if os.path.isdir(sub_dir_path):
            print(f"Processing directory: {sub_dir_path}")
            
            for dirpath, dirnames, filenames in os.walk(sub_dir_path, topdown=True):
                # Modify dirnames in-place to only keep the directories we want to retain
                dirnames[:] = [d for d in dirnames if d in keep_dirs]
                
                # Remove directories that are not in the keep_dirs list
                for dirname in os.listdir(dirpath):
                    full_path = os.path.join(dirpath, dirname)
                    if os.path.isdir(full_path) and dirname not in keep_dirs:
                        shutil.rmtree(full_path)
                        print(f"Removed directory: {full_path}")


def clean_save_dirs(root_dir):
    # Regex to match files it0 to it1100
    pattern = re.compile(r'^it([0-9]{1,4})-(train|[0-9]+)\.png$')
    
    for subdir, _, files in os.walk(root_dir):
        if os.path.basename(subdir) == "save":
            for file in files:
                if pattern.match(file):
                    file_path = os.path.join(subdir, file)
                    os.remove(file_path)
                    print(f"Removed file: {file_path}")
                    
                    
# Path to the root directory (update this with your actual path)
root_directory = "/home/ubuntu/scene/GaussianDreamer/outputs/MasterBedroom-33296-objects"  # Update with your actual path
# clean_gaussian_dreamer_dirs(root_directory)
clean_save_dirs(root_directory)