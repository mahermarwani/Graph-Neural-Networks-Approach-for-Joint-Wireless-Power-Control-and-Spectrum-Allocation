# Official repository for Graph Neural Networks Approach for Joint Wireless Power Control and Spectrum Allocation (The code will be uploaded shortly)

The proliferation of wireless technologies and the escalating performance requirements
of wireless applications have led to diverse and dynamic wireless environments, presenting formidable
challenges to existing radio resource management (RRM) frameworks. Researchers have proposed utilizing
deep learning (DL) models to address these challenges to learn patterns from wireless data and leverage
the extracted information to resolve multiple RRM tasks, such as channel allocation and power control.
However, it is noteworthy that the majority of existing DL architectures are designed to operate on
Euclidean data, thereby disregarding a substantial amount of information about the topological structure
of wireless networks. As a result, the performance of DL models may be suboptimal when applied to
wireless environments due to the failure to capture the network’s non-Euclidean geometry. This study
presents a novel approach to address the challenge of power control and spectrum allocation in an N-link
interference environment with shared channels, utilizing a graph neural network (GNN) based framework.
In this type of wireless environment, the available bandwidth can be divided into blocks, offering greater
flexibility in allocating bandwidth to communication links, but also requiring effective management of
interference. One potential solution to mitigate the impact of interference is to control the transmission
power of each link while ensuring the network’s data rate performance. Therefore, the power control
and spectrum allocation problems are inherently coupled and should be solved jointly. The proposed
GNN-based framework presents a promising avenue for tackling this complex challenge. Our experimental
results demonstrate that our proposed approach yields significant improvements compared to other existing
methods in terms of convergence, generalization, performance, and robustness, particularly in the context
of an imperfect channel.
