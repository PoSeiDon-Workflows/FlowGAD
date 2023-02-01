#!/usr/bin/env python3

import os
import subprocess

def check_if_empty_or_modified(arr):
    for k in range(len(arr)):
        if len(arr[k][1]) == 0:
            print(f"{arr[k][0]} has no timestamps")
        elif len(arr[k][1]) > 1:
            print(f"{arr[k][0]} has been modified {arr[k][1]}")
    return

def get_min_pos(arr):
    pos = 0
    for k in range(1, len(arr)):
        if arr[k][1] < arr[pos][1]:
            pos = k
    return pos

def get_git_times(arr):
    ret_arr = []
    for f in arr:
        process = subprocess.run(["git", "log", "--diff-filter=AM", "--format=%at", f],
                     stdout=subprocess.PIPE,
                     universal_newlines=True)
        add_modified_times = list(map(int, process.stdout.split()))
        add_modified_times.sort()
        ret_arr.append((f, add_modified_times))
    return ret_arr

file_dict = {}
keep_files = []
delete_files = []

for folder in os.listdir("."):
    if not os.path.isdir(folder):
        continue

    for filename in os.listdir(folder):
        abs_path = "/".join([folder, filename])
        if not os.path.isfile(abs_path):
            continue

        if filename in file_dict:
            file_dict[filename].append(abs_path)
        else:
            file_dict[filename] = [abs_path]

for filename in file_dict:
    if len(file_dict[filename]) > 1:
        add_modified_times = get_git_times(file_dict[filename])
        check_if_empty_or_modified(add_modified_times)
        #print(add_modified_times)
        min_pos = get_min_pos(add_modified_times)
        k_file = add_modified_times[min_pos][0]
        d_files = [add_modified_times[i][0] for i in range(len(add_modified_times)) if i != min_pos]
        keep_files.append(k_file)
        delete_files.extend(d_files)

print(f"keeping {len(keep_files)}")
print("\n".join(keep_files))
print(f"deleting {len(delete_files)}")
print("\n".join(delete_files))


#delete_files_list = " ".join(delete_files)
#print(f"git rm {delete_files_list}")
