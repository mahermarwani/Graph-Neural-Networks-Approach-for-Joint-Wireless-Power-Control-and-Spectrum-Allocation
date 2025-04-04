from wirelessNetwork import WirelessNetwork, calculate_network_metrics
import torch
import torch.nn.functional as F
import numpy as np
import dgl
from nets import Model
from tqdm import tqdm
from GA_solver import GA_solver
from utils import build_graph




def inference(model, csi, net_par, device, c_min, iterations=300, return_sol=False):
    """
    Run inference on the wireless network model.
    
    Args:
        model: The neural network model
        csi: Channel state information tensor
        net_par: Network parameters dictionary
        device: Computing device (CPU/GPU)
        c_min: Minimum capacity constraint
        iterations: Number of optimization iterations
        return_sol: Whether to return the solution or metrics
        
    Returns:
        Either (power allocation, resource block allocation) or
        (rates, rate history, constraint violation history)
    """
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
    g_list = tuple([g.to(device) for g in build_graph(csi)])
    csi = csi.to(device)

    rates_res = torch.zeros(csi.shape[1])
    rate_history = []
    violation_history = []
    min_rate_history = []
    max_rate_history = []
    std_rate_history = []

    # Create progress bar
    progress_bar = tqdm(total=iterations, desc="Iterations")

    for i in range(iterations):
        # Zero gradient for each iteration
        optimizer.zero_grad()

        # Model forward pass
        p_hat, rb_hat = model.forward(g_list)

        rb_hat = rb_hat.squeeze(0).t()
        p_hat = p_hat.squeeze(0)

        # Calculate network performance metrics
        rates = calculate_network_metrics(csi, rb_hat, p_hat, net_par, device)

        # Loss function: negative mean rate + penalty for constraint violations
        loss = -rates.mean() + torch.mean(F.relu(2 * c_min - rates)) - rates.min()
        loss.backward()

        # Update weights
        optimizer.step()

        # Convert soft resource block assignment to one-hot encoding
        true_rb = torch.eye(csi.shape[0], device=device)[torch.max(rb_hat, dim=1)[1]]

        # Calculate final rates with discrete resource block allocation
        rates_res = calculate_network_metrics(csi, true_rb, p_hat, net_par, device)
        
        # Calculate statistics
        mean_rate = rates_res.mean().item()
        min_rate = rates_res.min().item()
        max_rate = rates_res.max().item()
        std_rate = rates_res.std().item()
        violations = torch.sum(rates_res < c_min).item()

        # Update progress bar with enhanced statistics
        progress_bar.update(1)
        progress_bar.set_postfix({
            "Mean": f"{mean_rate:.2f}",
            "Min": f"{min_rate:.2f}",
            "Max": f"{max_rate:.2f}",
            "Std": f"{std_rate:.2f}",
            "Viol": violations
        })

        # Record history
        rate_history.append(mean_rate)
        violation_history.append(100 * (violations / csi.shape[1]))
        min_rate_history.append(min_rate)
        max_rate_history.append(max_rate)
        std_rate_history.append(std_rate)

    # Close the progress bar
    progress_bar.close()

    if return_sol:
        return p_hat, true_rb
    else:
        return rates_res, {
            'mean': rate_history,
            'min': min_rate_history,
            'max': max_rate_history,
            'std': std_rate_history,
            'violations': violation_history
        }


if __name__ == '__main__':
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

    # N = net_par['N']
    # K = net_par['K']

    # Create wireless network
    network = WirelessNetwork(net_par)
    # network.plot_network()  # Uncomment to visualize the network
    
    csi = network.csi

    # Set computing device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Working on:", device)

    # Load pre-trained model
    model = Model().to(device)
    model.load_state_dict(torch.load("final_model.pt", weights_only=True))

    # Run inference
    rates, history_data = inference(
        model=model,
        csi=csi,
        net_par=net_par,
        device=device,
        c_min=1e3,
        iterations=1000,
        return_sol=False
    )
    
    print("Final rates:", rates)
    print("\nRate history statistics:")
    print(f"  Mean rates: {history_data['mean'][-1]:.2f}")
    print(f"  Min rates: {history_data['min'][-1]:.2f}")
    print(f"  Max rates: {history_data['max'][-1]:.2f}")
    print(f"  Std rates: {history_data['std'][-1]:.2f}")
    print(f"  Violations (%): {history_data['violations'][-1]:.2f}%")