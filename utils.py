import torch
import dgl

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
