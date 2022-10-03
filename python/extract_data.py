import os


def get_subdirectories():
    subdirectories = []
    for directory in os.scandir():
        if directory.is_dir():
            path = os.path.join(os.getcwd(), directory.path)
            subdirectories.append(path)
    return subdirectories


def contains_tar_files(files):
    for file in files:
        if "tar.gz." in file:
            return True
    return False


def extract_tar_files():
    """ concatenate .tar.gz.aa, .tar.gz.ab,... files together to a .tar.gz file\n
     extract them into a folder\n
     delete the .tar.gz file"""

    if "full_archive.tar.gz" not in os.listdir():
        os.system("copy /b *.tar.gz.* full_archive.tar.gz")
    os.system("WinRAR X full_archive.tar.gz")
    os.system("del full_archive.tar.gz")


def move_folder_content():
    # could create problems if there are other subdirectories --> if len(getsubDirectories()) == 1
    os.chdir(get_subdirectories()[0])

    for folder in os.listdir():
        current_path = os.path.join(os.getcwd(), folder)
        destination = os.path.join(DATA_DIR_TARGET, folder)
        os.rename(current_path, destination)


# You need to create this subdirectory first, otherwise there might be problems
DATA_DIR_TARGET = os.path.join(os.getcwd(), "data/Coswara_processed/Recordings")
DATA_DIR_COSWARA = os.path.join(os.getcwd(), "data/Coswara-Data/")
os.chdir(DATA_DIR_COSWARA)

subdirs_coswara = get_subdirectories()

for subdir in subdirs_coswara:
    os.chdir(subdir)
    if contains_tar_files(os.listdir()):
        extract_tar_files()
        move_folder_content()
        print(os.getcwd())
