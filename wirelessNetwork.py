import random
import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.spatial import distance_matrix
import os
import warnings
from tqdm import tqdm

# Channel class for modeling wireless channel characteristics
# This class includes methods for calculating path loss, adding shadowing and fast fading effects.
class Channel:
    """Class for modeling wireless channel characteristics"""
    
    def __init__(self, htx, hrx, G, N0mdb, carrier_f, shadow_std=8):
        """
        Initialize Channel object
        
        Args:
            htx: Transmitter height (m)
            hrx: Receiver height (m)
            G: Antenna gain (dB)
            N0mdb: Noise density (mdB)
            carrier_f: Carrier frequency (Hz)
            shadow_std: Standard deviation for shadowing (dB)
        """
        self.htx = htx
        self.hrx = hrx
        self.G = G
        self.N0mdb = N0mdb
        self.N0 = np.power(10, ((self.N0mdb - 30) / 10))
        self.carrier_f = carrier_f
        self.shadow_std = shadow_std

    def path_loss(self, d):
        """
        Calculate path loss using an empirical model
        
        Args:
            d: Distance matrix between transmitters and receivers
            
        Returns:
            Path loss matrix
        """
        N = d.shape[0]
        signal_lambda = 2.998e8 / self.carrier_f
        
        # Compute relevant quantities
        Rbp = 4 * self.hrx * self.htx / signal_lambda
        Lbp = abs(20 * np.log10(np.power(signal_lambda, 2) / (8 * np.pi * self.htx * self.htx)))
        
        # Compute coefficient matrix for each Tx/Rx pair
        sum_term = 20 * np.log10(d / Rbp)
        Tx_over_Rx = Lbp + 6 + sum_term + ((d > Rbp).astype(int)) * sum_term  # Adjust for longer path loss
        
        # Add antenna gain
        if d.shape[1] == 1:  # For a vector of distances
            pathloss = -Tx_over_Rx + np.ones(N).reshape(d.shape) * self.G  # Only add antenna gain for direct channel
        else:  # For a matrix of distances
            pathloss = -Tx_over_Rx + np.eye(N) * self.G  # Only add antenna gain for direct channel
            
        pathloss = np.power(10, (pathloss / 10))  # Convert from decibel to absolute
        return pathloss

    def add_shadowing(self, channel_losses):
        """
        Add shadowing effects to channel losses
        
        Args:
            channel_losses: Matrix of channel losses
            
        Returns:
            Channel losses with shadowing
        """
        shadow_coefficients = np.random.normal(loc=0, scale=self.shadow_std, size=np.shape(channel_losses))
        channel_losses = channel_losses * np.power(10.0, shadow_coefficients / 10)
        return channel_losses

    def add_fast_fading(self, channel_losses):
        """
        Add fast fading effects to channel losses
        
        Args:
            channel_losses: Matrix of channel losses
            
        Returns:
            Tuple of (channel losses with fast fading, fast fading coefficients)
        """
        I = np.random.normal(loc=0, scale=1, size=np.shape(channel_losses))
        R = np.random.normal(loc=0, scale=1, size=np.shape(channel_losses))

        fastfadings = R + I * 1j
        channel_losses = channel_losses * (np.abs(fastfadings) ** 2) / 2
        
        return channel_losses, fastfadings

    def build_fading_capacity_channel(self, h, p):
        """
        Calculate capacity of a single user
        
        Args:
            h: Channel gain
            p: Transmit power
            
        Returns:
            Channel capacity
        """
        return np.log2(1 + h * p / self.N0 * 5e2)


# Define the WirelessNetwork class for modeling a wireless network
# This class includes methods for determining transmitter and receiver positions, generating channel state information.
class WirelessNetwork:
    """Class for modeling a wireless network"""
    
    def __init__(self, net_par):
        """
        Initialize WirelessNetwork object
        
        Args:
            net_par: Dictionary of network parameters
        """
        # Network parameters
        self.wx = net_par["wx"]  # Width of the network land
        self.wy = net_par["wy"]  # Length of the network land
        self.wc = net_par['wc']  # Maximum distance between users
        self.wd = net_par['wd']  # Minimum distance between users
        self.links_numb = net_par['N']  # Number of links

        # Determine transmitter and receiver positions
        self.t_pos, self.r_pos = self.determine_positions()

        # Calculate distance matrix using scipy.spatial method
        self.dist_mat = distance_matrix(self.t_pos, self.r_pos)
        
        # Channel parameters
        self.htx = net_par['htx']
        self.hrx = net_par['hrx']
        self.G = net_par['antenna_gain_decibel']
        self.N0mdb = net_par['noise_density_milli_decibel']
        self.N0 = np.power(10, ((self.N0mdb - 30) / 10))
        self.carrier_f = net_par['carrier_f']
        self.shadow_std = net_par['shadow_std']
        
        # Create channel with the given parameters
        self.channel = Channel(self.htx, self.hrx, self.G, self.N0mdb, self.carrier_f, self.shadow_std)

        # Resource blocks parameters
        self.rb_bandwidth = net_par['rb_bandwidth']
        self.rb_numb = net_par['K']

        # Generate channel state information tensor [number of blocks, number of Tx, number of Rx]
        self.csi, self.fast_fading = self.generate_csi()

    def determine_positions(self):
        """
        Determine transmitter and receiver positions
        
        Returns:
            Tuple of (transmitter positions, receiver positions)
        """
        # Calculate transmitter positions
        t_x_pos = np.random.uniform(0, self.wx, (self.links_numb, 1))
        t_y_pos = np.random.uniform(0, self.wy, (self.links_numb, 1))
        t_pos = np.hstack((t_x_pos, t_y_pos))
        
        # Calculate receiver positions
        r_distance = np.random.uniform(self.wd, self.wc, (self.links_numb, 1))
        r_angle = np.random.uniform(0, 2 * np.pi, (self.links_numb, 1))
        r_rel_pos = r_distance * np.hstack((np.cos(r_angle), np.sin(r_angle)))
        r_pos = t_pos + r_rel_pos
        
        return t_pos, r_pos 

    def plot_network(self):
        """Plot transmitter and receiver positions"""
        plt.figure()
        plt.scatter(self.t_pos[:, 0], self.t_pos[:, 1], marker='o', color='b')
        plt.scatter(self.r_pos[:, 0], self.r_pos[:, 1], marker='x', color='r')

        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(['Transmitter', 'Receiver'])
        plt.grid(True)
        plt.show()

    def generate_csi(self):
        """
        Generate channel state information
        
        Returns:
            Tuple of (channel state information, fast fading coefficients)
        """
        channel_gain = np.array([self.channel.path_loss(self.dist_mat) for _ in range(self.rb_numb)]) 
        channel_gain = self.channel.add_shadowing(channel_gain)
        channel_losses, fast_fadings = self.channel.add_fast_fading(channel_gain)

        return torch.from_numpy(channel_losses).float(), torch.from_numpy(fast_fadings)


# Calculate network performance metrics
# This function computes the achievable rates for each user based on the channel state information, resource block allocation, and transmit power.
def calculate_network_metrics(csi, rb, p, net_par, device="cpu"):
    """
    Calculate network performance metrics
    
    Args:
        csi: Channel state information
        rb: Resource block allocation
        p: Transmit power
        net_par: Network parameters
        device: Computation device
        
    Returns:
        Achievable rates for each user
    """
    K = net_par["K"]  # Number of resource blocks
    N = net_par["N"]  # Number of links

    N0 = np.power(10, ((net_par["noise_density_milli_decibel"] - 30) / 10))
    bandwidth = net_par["rb_bandwidth"]

    rates = torch.zeros(N, device=device)
    for k in range(K):
        received_power = torch.diag(csi[k, :, :]) * rb[:, k] * p
        interference = torch.matmul(csi[k, :, :], rb[:, k] * p) - received_power

        noise_power = N0 * bandwidth

        snr = received_power / (interference + noise_power)
        snr = torch.clamp(snr, min=0.0)

        rates = torch.add(rates, bandwidth * rb[:, k] * torch.log2(1 + snr))
        
    return rates

# Calculate energy efficiency
# This function computes the energy efficiency for each user based on the achievable rates and transmit power.
def calculate_energy_efficiency(rates, power, device="cpu"):
    """
    Calculate energy efficiency
    
    Args:
        rates: Achievable rates
        power: Transmit power
        device: Computation device
        
    Returns:
        Energy efficiency for each user
    """
    # Avoid division by zero
    safe_power = torch.maximum(power, torch.tensor(1e-10, device=device))
    return rates / safe_power

if __name__ == '__main__':
    # Set computation device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    # Define network parameters
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

    # Extract key parameters
    N, K = net_par['N'], net_par['K']

    # Create wireless network and get channel state information
    network = WirelessNetwork(net_par)
    csi = network.csi.to(device)

    # Initialize and allocate resource blocks and power
    # Each user gets one random resource block and random power
    rb = torch.zeros((N, K), device=device)
    random_rbs = torch.randint(0, K, (N,), device=device)
    rb[torch.arange(N, device=device), random_rbs] = 1
    p = torch.rand(N, device=device)

    # Display input parameters
    print(f"Using device: {device}")
    print("\nInput parameters:")
    print(f"CSI shape: {csi.shape}")
    print(f"RB allocation shape: {rb.shape}")
    print(f"Power allocation shape: {p.shape}")

    # Calculate network performance metrics
    rates = calculate_network_metrics(csi, rb, p, net_par, device=device)
    energy_efficiency = calculate_energy_efficiency(rates, p, device=device)

    # Display results
    print("\nOutput results:")
    print(f"Average rate: {rates.mean().item():.3f} bps")
    print(f"Min/Max rates: {rates.min().item():.3f}/{rates.max().item():.3f} bps")
    print(f"Average energy efficiency: {energy_efficiency.mean().item():.3f}")
    print(f"Total network throughput: {rates.sum().item():.3f} bps")