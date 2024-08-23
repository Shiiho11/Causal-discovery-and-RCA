import json
import os

import pandas as pd


def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"The folder '{directory}' has been successfully created!")
    else:
        print(f"The folder '{directory}' already exists.")
