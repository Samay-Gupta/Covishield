import os

cur_dir = os.getcwd()
storage_dir = os.path.join(cur_dir, "storage")
static_data_file = os.path.join(cur_dir, "modules", "static_data.py")
with open(static_data_file) as f:
    filelines = f.readlines()
    filelines[0] = f'STORAGE_PATH = "{storage_dir}"'
    with open(static_data_file, 'w') as f:
        f.writelines(filelines)

os.system("pip install -r requirements.txt")