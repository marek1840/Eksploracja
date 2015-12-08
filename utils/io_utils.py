import os


def check_if_dir_exists_else_create(path):
    if not os.path.exists(path):
        os.makedirs(path)
