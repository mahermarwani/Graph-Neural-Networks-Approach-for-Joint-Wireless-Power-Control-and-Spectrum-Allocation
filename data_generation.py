import csv
import os
import sys
import numpy as np
from pymoo.core.mixed import MixedVariableGA
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Integer
from pymoo.core.variable import Real
from pymoo.optimize import minimize as pyminimize
from tqdm import tqdm

from wirelessNetwork import WirelessNetwork
from GA_solver import GA_solver


def data_generator(net_par, 
                    length, 
                    max_power, 
                    c_min, 
                    dataset_name=None):
    if dataset_name is None:
        dataset_name = 'data_' + str(length) + 'p_max_' + str(max_power)

    path = os.path.join(dataset_name)

    # Create folders
    path_csi = os.path.join(path, 'csi')
    path_p = os.path.join(path, 'p')
    path_rb = os.path.join(path, 'rb')
    path_csv = os.path.join(path, 'samples_list.csv')

    if os.path.exists(path):
        print("dataset folder name already exist...!")
        return None
    else:
        os.makedirs(path_csi)
        os.makedirs(path_p)
        os.makedirs(path_rb)
        with open(path_csv, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["csi ID", "power ID", "rb ID"])
        file.close()

    K = net_par["K"]
    N = net_par["N"]


    for i in tqdm(range(length)):
        # Generate random channel gain
        network = WirelessNetwork(net_par)
        
        p, rb = GA_solver(network.csi, net_par, eval=20000, c_min=c_min)

        np.save(os.path.join(path_csi, "csi_" + str(i) + ".npy"), network.csi)
        np.save(os.path.join(path_p, "p_" + str(i) + ".npy"), p)
        np.save(os.path.join(path_rb, "rb_" + str(i) + ".npy"), rb)
        with open(path_csv, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["csi_" + str(i), "p_" + str(i), "rb_" + str(i)])
        file.close()





if __name__ == '__main__':
    net_par = {
        "d0": 1,
        'htx': 1.5,
        'hrx': 1.5,
        'antenna_gain_decibel': 2.5,
        'noise_density_milli_decibel': -169,
        'carrier_f': 2.4e9,
        'shadow_std': 8,
        "rb_bandwidth": 5e2,
        "wc": 50,
        "wd": 20,
        "wx": 500,
        "wy": 500,
        "N": 50,  # Number of links
        "K": 10    # Number of resource blocks
    }

    print("Generating dataset...")
    length = 3
    max_power = 1
    c_min = 1e3
    dataset_name = "DATASET_train_{}_p_{}c={}_N={}_rb=10".format(length, max_power, c_min, net_par["N"], net_par["K"])
    data_generator(net_par, length=length,
                      max_power=max_power,
                      c_min=c_min,
                      dataset_name=dataset_name)
