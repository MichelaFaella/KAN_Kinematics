import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np

class DataLoader:
    def __init__(self):
        self.data = {}

    # Load data from pickle files based on the specified stage, deformation type, and trial number
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
                        # Concatenate data if the key already exists, otherwise initialize it
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
        # Return the keys of the loaded data
        return self.data.keys()

    def get_data(self):
        # Return the loaded data
        return self.data
    
    def get_num_samples(self):
        # Return the number of samples in the first data entry
        first_key = next(iter(self.data))
        return self.data[first_key].shape[0]
        
if __name__ == "__main__":
    dm = DataLoader()
    dm.load_data(deformation="bending", trial_num=1)
    import matplotlib.pyplot as plt

    data = dm.get_data()
    actuation = data["actuation"][:, 2]
    plt.plot(actuation, label="Line")
    plt.scatter(np.arange(len(actuation)), actuation, color='red', s=10, label="Values")
    plt.legend()
    plt.title("Actuation Data (Column 2)")
    plt.xlabel("Sample Index")
    plt.ylabel("Actuation Value")
    plt.show()
    print("Position: ",dm.get_data()["markers"][0,-1,:])