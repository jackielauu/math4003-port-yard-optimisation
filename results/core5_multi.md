| Instance | Gurobi status | Gurobi obj | Gurobi gap | Gurobi time | Benders status | Benders obj | Benders gap | Benders time | Benders cuts | Faster / better |
|---|---|---|---|---|---|---|---|---|---|---|
| F1_system_scale_H seed 1 | OPTIMAL | 28,340,322 | 0.040% | 55s | TIME_LIMIT | 26,300,802 | 37.175% | 600s | 141 feas / 3 opt (144 iters) | Gurobi |
| F1_system_scale_L seed 1 | OPTIMAL | 12,690,204 | 0.000% | 2s | TIME_LIMIT | 12,552,684 | 12.326% | 600s | 0 feas / 27 opt (27 iters) | Gurobi |
| F1_system_scale_M seed 1 | OPTIMAL | 19,418,604 | 0.095% | 7s | TIME_LIMIT | 18,632,844 | 22.688% | 601s | 9 feas / 12 opt (21 iters) | Gurobi |
| F2_mode_tradeoff_H seed 1 | OPTIMAL | 17,273,733 | 0.028% | 11s | TIME_LIMIT | 16,883,733 | 27.473% | 601s | 189 feas / 4 opt (193 iters) | Gurobi |
| F4_operational_conflict_H seed 1 | OPTIMAL | 18,914,529 | 0.095% | 17s | TIME_LIMIT | 18,208,689 | 23.192% | 601s | 21 feas / 18 opt (39 iters) | Gurobi |

Instances: 5. Solved to the gap target: Gurobi 5, Benders 0. Wins: Gurobi 5, Benders 0.
Total Benders cuts across all runs: 360 feasibility, 64 optimality.
