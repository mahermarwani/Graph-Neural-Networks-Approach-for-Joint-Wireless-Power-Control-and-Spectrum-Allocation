from wirelessNetwork import WirelessNetwork, calculate_network_metrics
import torch
import torch.nn.functional as F
import numpy as np
import dgl
from nets import Model
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


def build_graph(csi):
    """
    Build a list of DGL graphs from channel state information.
    
    Args:
        csi: Channel state information tensor
        
    Returns:
        List of DGL graphs representing network connections
    """
    K = csi.shape[0]  # number of resource blocks
    N = csi.shape[1]  # number of users
    
    # Create adjacency list (complete graph without self-loops)
    adj = [(i, j) for i in range(N) for j in range(N) if i != j]
    src, dst = zip(*adj)
    src = torch.tensor(src)
    dst = torch.tensor(dst)

    # Normalize CSI data
    log_x = torch.log10(csi)
    mean_log_x = torch.mean(log_x)
    variance_log_x = torch.var(log_x)
    csi = (log_x - mean_log_x) / torch.sqrt(variance_log_x)

    # Build a graph for each Resource Block
    g_list = []
    for k in range(K):
        # Create a graph
        graph = dgl.graph((src, dst), num_nodes=N)

        # Node feature: direct link channel of i-th pair
        graph.ndata['feature'] = torch.diag(csi[k]).unsqueeze(1)

        # Edge feature: interference channel between pairs
        graph.edata['feature'] = torch.stack([
            csi[k, src, dst],  # Features from source to destination
            csi[k, dst, src]   # Features from destination to source
        ], dim=1)

        g_list.append(graph)

    return g_list


def inference_with_history(model, csi, net_par, device, c_min, iterations=300, sample_freq=5):
    """
    Run inference on the wireless network model and collect detailed rate history.
    
    Args:
        model: The neural network model
        csi: Channel state information tensor
        net_par: Network parameters dictionary
        device: Computing device (CPU/GPU)
        c_min: Minimum capacity constraint
        iterations: Number of optimization iterations
        sample_freq: How often to collect full distribution data
        
    Returns:
        Distribution history and summary statistics
    """
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
    g_list = tuple([g.to(device) for g in build_graph(csi)])
    csi = csi.to(device)

    # History containers
    rate_history = []
    violation_history = []
    min_rate_history = []
    max_rate_history = []
    std_rate_history = []
    
    # Full distribution history (sampled less frequently)
    distribution_history = []
    iteration_markers = []

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
        
        # Save full distribution periodically
        if i % sample_freq == 0 or i == iterations - 1:
            distribution_history.append(rates_res.detach().cpu().numpy())
            iteration_markers.append(i)

    # Close the progress bar
    progress_bar.close()

    history_data = {
        'mean': rate_history,
        'min': min_rate_history,
        'max': max_rate_history,
        'std': std_rate_history,
        'violations': violation_history,
        'distributions': distribution_history,
        'iterations': iteration_markers,
        'c_min': c_min
    }

    return rates_res, history_data


def plot_rate_distributions(history_data, save_path="rate_distributions.png"):
    """
    Create a static plot showing the evolution of rate distributions at key iterations.
    
    Args:
        history_data: Dictionary containing distribution history and iteration markers
        save_path: Where to save the plot
    """
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Select iterations to display (first, some middle points, and last)
    num_frames = len(history_data['distributions'])
    if num_frames <= 5:
        indices = list(range(num_frames))
    else:
        indices = [0]  # First
        interval = (num_frames - 2) // 3
        indices.extend([interval, 2*interval, num_frames-1])  # Some middle points and last
    
    # Create subplots
    fig, axes = plt.subplots(len(indices), 1, figsize=(10, 3*len(indices)))
    if len(indices) == 1:
        axes = [axes]
    
    fig.suptitle('Evolution of User Rate Distribution', fontsize=16)
    
    for i, idx in enumerate(indices):
        iter_num = history_data['iterations'][idx]
        distribution = history_data['distributions'][idx]
        
        # Create violin plot
        sns.violinplot(data=[distribution], ax=axes[i], palette="Blues")
        
        # Add statistics as text
        stats_str = (
            f"Mean: {history_data['mean'][iter_num]:.2f}, "
            f"Min: {history_data['min'][iter_num]:.2f}, "
            f"Max: {history_data['max'][iter_num]:.2f}, "
            f"Std Dev: {history_data['std'][iter_num]:.2f}, "
            f"Violations: {history_data['violations'][iter_num]:.1f}%"
        )
        axes[i].text(
            0.5, 0.9, stats_str, 
            transform=axes[i].transAxes,
            horizontalalignment='center',
            bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.7}
        )
        
        # Add constraint line
        axes[i].axhline(y=history_data['c_min'], color='r', linestyle='--', alpha=0.7, label='Min Rate Constraint')
        
        # Set labels
        axes[i].set_title(f'Rate Distribution at Iteration {iter_num}')
        axes[i].set_ylabel('User Rate')
        axes[i].set_xticks([])
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Distribution plot saved to {save_path}")
    return save_path


def plot_statistics_evolution(history_data, save_path="rate_statistics.png"):
    """
    Create a plot showing the evolution of rate statistics over iterations.
    
    Args:
        history_data: Dictionary containing statistics history
        save_path: Where to save the plot
    """
    plt.figure(figsize=(12, 8))
    plt.style.use('seaborn-v0_8-darkgrid')
    
    iterations = range(len(history_data['mean']))
    
    # Plot rate statistics
    plt.subplot(2, 1, 1)
    plt.plot(iterations, history_data['mean'], label='Mean Rate', linewidth=2)
    plt.plot(iterations, history_data['min'], label='Min Rate', linewidth=2)
    plt.plot(iterations, history_data['max'], label='Max Rate', linewidth=2)
    plt.plot(iterations, history_data['std'], label='Std Dev', linewidth=2)
    plt.axhline(y=history_data['c_min'], color='r', linestyle='--', 
                label='Min Rate Constraint', linewidth=1.5)
    
    plt.title('Evolution of Rate Statistics', fontsize=14)
    plt.xlabel('Iteration')
    plt.ylabel('Rate Values')
    plt.legend()
    plt.grid(True)
    
    # Plot violations percentage
    plt.subplot(2, 1, 2)
    plt.plot(iterations, history_data['violations'], color='red', linewidth=2)
    plt.title('Constraint Violations Over Time', fontsize=14)
    plt.xlabel('Iteration')
    plt.ylabel('Violations (%)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Statistics plot saved to {save_path}")
    return save_path



def create_simple_animation(history_data, save_path="violin_plot_animation.gif", dark_mode=False, dpi=300):
    """
    Create a professional animation of network performance evolution using only violin plots.
    
    Args:
        history_data: Dictionary containing distribution history data with keys:
                     'iterations', 'distributions', 'mean', 'min', 'max', 
                     'std', 'violations', 'c_min'
        save_path: Where to save the GIF animation
        dark_mode: Whether to use dark background theme (True) or light theme (False)
    """
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    from scipy.stats import gaussian_kde
    
    # Set appropriate colors based on theme
    if dark_mode:
        plt.style.use('dark_background')
        text_color = 'white'
        background_color = '#121212'
        panel_color = '#1E1E1E'
        grid_color = '#333333'
    else:
        plt.style.use('default')
        text_color = '#333333'
        background_color = '#FFFFFF'
        panel_color = '#F5F5F5'
        grid_color = '#DDDDDD'
    
    # Define color palette
    highlight_color = '#2196F3'  # Blue
    accent_colors = ['#4CAF50', '#FF9800', '#E91E63', '#9C27B0', '#F44336']  # Green, Orange, Pink, Purple, Red
    box_alpha = 0.2 if not dark_mode else 0.7
        
    # Create figure with single panel
    fig = plt.figure(figsize=(10, 8), dpi=dpi, facecolor=background_color)
    fig.suptitle('Network Performance Optimization Over Iterations', fontsize=20, color=text_color)
    
    # Initialize panel
    ax_violin = plt.gca()
    ax_violin.set_facecolor(panel_color)
    
    # Extract data
    iterations = history_data['iterations']
    max_iterations = max(iterations)
    
    # Pre-compute colors for users based on initial rates
    first_rates = history_data['distributions'][0]
    rate_colors = plt.cm.coolwarm(
        (first_rates - np.min(first_rates)) / (np.max(first_rates) - np.min(first_rates))
    )
    
    # Style the axes
    for spine in ax_violin.spines.values():
        spine.set_color(grid_color)
    ax_violin.tick_params(colors=text_color, which='both')
    
    # Add copyright info
    # fig.text(0.01, 0.01, "© Network Optimization Team", fontsize=10, color=text_color, alpha=0.7)
    
    
    # Function to update the figure for each frame
    def update(frame):
        # Get current iteration data
        iteration_idx = frame
        iteration = iterations[iteration_idx]
        distribution = history_data['distributions'][iteration_idx]
        
        # Update main violin plot
        ax_violin.clear()
        
        # Create violin plot
        parts = ax_violin.violinplot(
            [distribution], positions=[0], showmeans=False, showmedians=False,
            widths=0.6
        )
        
        # Style violin
        for pc in parts['bodies']:
            pc.set_facecolor(highlight_color)
            pc.set_edgecolor(text_color)
            pc.set_alpha(0.7)
        
        # Add scatter points with jitter
        jitter = np.random.normal(0, 0.05, size=len(distribution))
        ax_violin.scatter(jitter, distribution, alpha=0.6, s=15, c=rate_colors, zorder=3)
        
        # Add box plot inside violin
        box_parts = ax_violin.boxplot(
            [distribution], positions=[0], widths=0.15, 
            patch_artist=True, showfliers=False, zorder=4
        )
        
        # Style box plot elements
        for box in box_parts['boxes']:
            box.set(facecolor=accent_colors[0], alpha=0.7)
            box.set(edgecolor=text_color)
        
        for whisker in box_parts['whiskers']:
            whisker.set(color=text_color, linewidth=1.5)
            
        for cap in box_parts['caps']:
            cap.set(color=text_color, linewidth=1.5)
            
        for median in box_parts['medians']:
            median.set(color=accent_colors[4], linewidth=2)
        

        # Add constraint line
        constraint_line = ax_violin.axhline(
            y=history_data['c_min'], color=accent_colors[2], linestyle='--', 
            linewidth=2, alpha=0.8, label='Min Rate Constraint'
        )
        
        # Calculate improvement percentages for non-initial frames
        if iteration_idx > 0:
            first_iter = iterations[0]
            mean_change = ((history_data['mean'][iteration] / history_data['mean'][first_iter]) - 1) * 100
            min_change = ((history_data['min'][iteration] / history_data['min'][first_iter]) - 1) * 100
            max_change = ((history_data['max'][iteration] / history_data['max'][first_iter]) - 1) * 100
            violation_change = (
                (history_data['violations'][iteration] / history_data['violations'][first_iter]) - 1
            ) * 100
            
            change_text = (
                f"Mean: {'+' if mean_change >= 0 else ''}{mean_change:.1f}%\n"
                f"Min: {'+' if min_change >= 0 else ''}{min_change:.1f}%\n"
                f"Max: {'+' if max_change >= 0 else ''}{max_change:.1f}%"
                f"\nViolations: {'+' if violation_change >= 0 else ''}{violation_change:.1f}%"
            )
        else:
            change_text = "Baseline"
        
        # Add status text that dynamically updates based on the iteration index.
        # It displays different messages for the start, optimization process, and completion.
        if iteration_idx == 0:
            status_text = "▶ Starting..."
        elif iteration_idx == len(iterations) - 1:
            status_text = "✓ Optimization Complete"
        else:
            status_text = "⟳ Optimizing..."

        status_color = accent_colors[0] if (iteration_idx == len(iterations) - 1 or iteration_idx == 0) else highlight_color
        
        # Current statistics text
        stats_str = (
            f"Iteration: {iteration}\n"
            f"Mean: {history_data['mean'][iteration]:.2f} Kbps\n"
            f"Min: {history_data['min'][iteration]:.2f} Kbps\n"
            f"Max: {history_data['max'][iteration]:.2f} Kbps\n"
            f"Std Dev: {history_data['std'][iteration]:.2f}\n"
            f"Violations: {history_data['violations'][iteration]:.1f}%"
        )
        
        # Add stats text boxes
        ax_violin.text(
            0.05, 0.95, stats_str,
            transform=ax_violin.transAxes,
            horizontalalignment='left',
            verticalalignment='top',
            bbox={'boxstyle': 'round,pad=0.5', 'facecolor': panel_color, 'edgecolor': grid_color, 'alpha': box_alpha},
            color=text_color,
            fontsize=10
        )
        
        ax_violin.text(
            0.95, 0.95, change_text,
            transform=ax_violin.transAxes,
            horizontalalignment='right',
            verticalalignment='top',
            bbox={'boxstyle': 'round,pad=0.5', 'facecolor': panel_color, 'edgecolor': grid_color, 'alpha': box_alpha},
            color=text_color,
            fontsize=10
        )
        
        ax_violin.text(
            0.5, 0.98, status_text,  # Adjusted y-coordinate to move above the plot
            transform=ax_violin.transAxes,
            horizontalalignment='center',
            verticalalignment='top',
            bbox={'boxstyle': 'round,pad=0.5', 'facecolor': panel_color, 'edgecolor': status_color, 'alpha': box_alpha},
            color=status_color,
            fontsize=20,
            fontweight='bold'
        )
        
        # Style the violin plot

        ax_violin.set_ylabel('Rate (×10³ Kbps)', fontsize=12, color=text_color)

        ax_violin.set_xticks([])
        ax_violin.grid(True, linestyle='--', alpha=0.3, color=grid_color)
        
        # Set y-axis limits and tick colors
        y_min = max(0, min(min(distribution), history_data['c_min']) * 0.9)
        y_max = max(distribution) * 1.1
        ax_violin.set_ylim(y_min, y_max)
        ax_violin.tick_params(colors=text_color)

        import matplotlib.ticker as ticker

        # Automatically scale the tick labels
        def format_ticks(value, _):
            return f'{int(value / 1000):02d}'  # E.g. 01, 02, 03...

        ax_violin.yaxis.set_major_formatter(ticker.FuncFormatter(format_ticks))

        
        # Add legend
        # Create legend handles for all elements
        scatter_leg = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=highlight_color, 
                     markersize=8, alpha=0.6)
        box_leg = plt.Rectangle((0,0), 1, 1, facecolor=accent_colors[0], alpha=0.7, 
                    edgecolor=text_color)
        
        # Add legend with all elements
        ax_violin.legend([parts['bodies'][0], box_leg, box_parts['medians'][0], scatter_leg, constraint_line], 
               ['Rate Distribution', 'Rate Quartiles', 'Median Rate', 'Individual Users', 'Min Rate Requirement'],
               loc='center right', frameon=True, facecolor=panel_color, framealpha=0.8, edgecolor=grid_color)
        

        
        return (constraint_line)
    
    # Create and save animation with pauses at the beginning and end
    frames = len(history_data['distributions'])
    pause_frames = 10  # Number of frames to pause at the start and end

    def extended_update(frame):
        if frame < pause_frames:
            return update(0)  # Pause at the first frame
        elif frame >= frames + pause_frames:
            return update(frames - 1)  # Pause at the last frame
        else:
            return update(frame - pause_frames)

    ani = animation.FuncAnimation(
        fig, extended_update, frames=frames + 2 * pause_frames, interval=200, blit=False
    )
    # Adjust layout and save
    plt.tight_layout(rect=[0.03, 0, 1, 1])
    ani.save(save_path, writer='pillow', fps=60, dpi=100)
    plt.close()
    
    print(f"Violin plot animation saved to {save_path}")
    return save_path








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
    csi = network.csi

    # Set computing device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Working on:", device)

    # Load pre-trained model
    model = Model().to(device)
    model.load_state_dict(torch.load("final_model.pt", weights_only=True))

    # Run inference with detailed history tracking
    rates, history_data = inference_with_history(
        model=model,
        csi=csi,
        net_par=net_par,
        device=device,
        c_min=1e3,
        iterations=500,
        sample_freq=1  # Collect distribution data every 5 iterations
    )
    
    # Try to create animation first (GIF format to avoid ffmpeg issues)
    animation_path = create_simple_animation(history_data, save_path="rate_animation.gif")
    
    # Always create static plots as fallback
    dist_plot_path = plot_rate_distributions(history_data)
    stats_plot_path = plot_statistics_evolution(history_data)
    
    print("\nVisualization complete!")
    print("\nFinal statistics:")
    print(f"  Mean rate: {history_data['mean'][-1]:.2f}")
    print(f"  Min rate: {history_data['min'][-1]:.2f}")
    print(f"  Max rate: {history_data['max'][-1]:.2f}")
    print(f"  Std dev: {history_data['std'][-1]:.2f}")
    print(f"  Violations: {history_data['violations'][-1]:.2f}%")