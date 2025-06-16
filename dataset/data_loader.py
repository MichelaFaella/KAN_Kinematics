import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np


class DataLoader:
    def __init__(self):
        self.data = {}

    # Load dataset from pickle files based on the specified stage, deformation type, and trial number
    # deformation: "bending" (default), "twisting_CW", "twisting_CCW" or "all"
    # trial_num: 0 (all, default), 1 or more
    def load_data(self, deformation="bending", trial_num=0):
        if trial_num == 0:
            # Load all trials for the specified stage and deformation
            file_path = f'dataset/data/{deformation}'
            for file_name in os.listdir(file_path):
                if file_name.endswith('.pkl'):
                    with open(os.path.join(file_path, file_name), 'rb') as file:
                        data = pickle.load(file)
                        # Concatenate dataset if the key already exists, otherwise initialize it
                        for key, value in data.items():
                            if key in self.data:
                                self.data[key] = np.concatenate((self.data[key], value), axis=0)
                            else:
                                self.data[key] = value
        else:
            # Load a specific trial
            file_path = f'dataset/data/{deformation}/trial_{trial_num}.pkl'
            with open(file_path, 'rb') as file:
                self.data = pickle.load(file)

    def get_keys(self):
        # Return the keys of the loaded dataset
        return self.data.keys()

    def get_data(self):
        # Return the loaded dataset
        return self.data

    def get_num_samples(self):
        # Return the number of samples in the first dataset entry
        first_key = next(iter(self.data))
        return self.data[first_key].shape[0]


if __name__ == "__main__":
    dm = DataLoader()
    dm.load_data(deformation="bending", trial_num=1)
    data = dm.get_data()
    print("Actuation: ", dm.get_data()["actuation"][0])
    print("Position: ", dm.get_data()["markers"][0, -1, :])
