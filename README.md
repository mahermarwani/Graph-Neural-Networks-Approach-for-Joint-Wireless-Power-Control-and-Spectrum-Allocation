# Official repository for Graph Neural Networks Approach for Joint Wireless Power Control and Spectrum Allocation 

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


## Results:
### Simulation Results

#### Scenario: 500x500 Area with 50 D2D Links
The simulation was conducted in a 500x500 area with 50 Device-to-Device (D2D) communication links. The minimum data rate constraint (`c_min`) was set to 1e3, and the available bandwidth was divided into 10 Resource Blocks (RBs) of 500Hz each.

#### Visualization
The following animation illustrates the rate performance over time during the simulation:

![Rate Animation](rate_animation.gif)

This visualization highlights the dynamic behavior of the proposed GNN-based framework in managing power control and spectrum allocation effectively.


## Installation Instructions

To set up the environment and get started with the project, follow these steps:

1. **Clone the Repository**:  
    Clone this repository to your local machine using the following command:  
    ```bash
    git clone https://github.com/mahermarwani/Graph-Neural-Networks-Approach-for-Joint-Wireless-Power-Control-and-Spectrum-Allocation.git
    cd your-repo-name
    ```

2. **Install Dependencies**:  
    Install the required dependencies listed in the `requirements.txt` file:  
    ```bash
    pip install -r requirements.txt
    ```

You are now ready to proceed with the steps outlined above to generate data, train the model, and evaluate the results.

To get started with the codebase, follow these steps:

1. **Generate Wireless Networks**:  
    Use the `wireless.Network.py` script to create wireless network topologies.  
    ```bash
    python wireless/Network.py
    ```

2. **Generate Data**:  
    Use the `data_generation.py` script to generate the required dataset for training and testing.  
    ```bash
    python data_generation.py
    ```

3. **Train the GNN Model**:  
    Train the Graph Neural Network model using the `training.py` script.  
    ```bash
    python training.py
    ```

4. **Test the Model**:  
    Evaluate the trained model using the `testing.py` script.  
    ```bash
    python testing.py
    ```

5. **Solve Using Genetic Algorithm**:  
    Alternatively, solve the problem using the Genetic Algorithm approach with the `GA_solver.py` script.  
    ```bash
    python GA_solver.py
    ```

Ensure all dependencies are installed before running the scripts. Refer to the `requirements.txt` file for the list of dependencies.


