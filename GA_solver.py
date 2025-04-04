import torch
import numpy as np
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from pymoo.core.mixed import MixedVariableGA
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Integer
from pymoo.optimize import minimize
from wirelessNetwork import WirelessNetwork, calculate_network_metrics


class MINLP(ElementwiseProblem):

    def __init__(self, csi, c_min, net_par, **kwargs):
        self.net_par = net_par
        self.csi = csi
        self.c_min = c_min

        vars = {}
        for i in range(net_par["N"]):
            vars[f"rb_{i}"] = Integer(bounds=(0, net_par["K"] - 1))
            vars[f"p_{i}"] = Real(bounds=(0, 1))

        super().__init__(vars=vars, n_obj=1, n_ieq_constr=net_par["N"], **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        rb, p = [], []

        for key, val in X.items():
            if key.startswith("rb"):
                rb.append(val)
            elif key.startswith("p"):
                p.append(val)

        rb = torch.eye(self.net_par["K"])[torch.tensor(rb).long()].float()
        p = torch.tensor(p).float()

        rates = calculate_network_metrics(self.csi, rb, p, self.net_par)
        out["G"] = [(self.c_min - rate.item()) for rate in rates]
        out["F"] = [-rates.mean().item()]


def GA_solver(csi, net_par, eval=20000, c_min=300):
    problem = MINLP(csi, c_min, net_par)
    algorithm = MixedVariableGA(pop=1000, survival=RankAndCrowdingSurvival())

    res = minimize(problem, algorithm, termination=('n_evals', eval), verbose=True)



    if res.F is None:
        print("No feasible solution found. Returning solution with least constraint violations.")
        best_violations = float('inf')
        best_rates = None
        best_p, best_rb = None, None

        for ind in res.pop:
            p, rb = [], []

            for key, val in ind.X.items():
                if key.startswith("rb"):
                    rb.append(val)
                elif key.startswith("p"):
                    p.append(val)

            rb_tensor = torch.eye(net_par["K"])[torch.tensor(rb).long()].float()
            p_tensor = torch.tensor(p).float()

            rates = calculate_network_metrics(csi, rb_tensor, p_tensor, net_par)
            out = {}
            problem._evaluate(ind.X, out)

            violations = sum(g > 0 for g in out["G"])

            if violations < best_violations:
                best_violations = violations
                best_rates = rates
                best_p, best_rb = p_tensor, rb_tensor

        return best_p, best_rb

    else:
        print("Feasible solution found.")
        p, rb = [], []

        for key, val in res.X.items():
            if key.startswith("rb"):
                rb.append(val)
            elif key.startswith("p"):
                p.append(val)

        rb_tensor = torch.eye(net_par["K"])[torch.tensor(rb).long()].float()
        p_tensor = torch.tensor(p).float()

        return p_tensor, rb_tensor


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
        "K": 10   # Number of resource blocks
    }

    network = WirelessNetwork(net_par)
    p, rb = GA_solver(network.csi, net_par, eval=20000, c_min=1000)

    rates = calculate_network_metrics(network.csi, rb, p, net_par)

    print("p:", p)
    print("rb indices:", torch.where(rb != 0)[1])
    print("Rates:", rates)
    print("Mean rate:", rates.mean())
    print("Violations (rates < 100):", sum(rate.item() < 1000 for rate in rates))
