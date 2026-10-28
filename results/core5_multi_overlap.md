| Instance | Gurobi status | Gurobi obj | Gurobi gap | Gurobi time | Benders status | Benders obj | Benders gap | Benders time | Benders cuts | Faster / better |
|---|---|---|---|---|---|---|---|---|---|---|
| F1_system_scale_H seed 1 | OPTIMAL | 28,421,682 | 0.033% | 77s | TIME_LIMIT | 25,970,322 | 40.920% | 601s | 376 feas / 3 opt (379 iters) | Gurobi |
| F1_system_scale_L seed 1 | OPTIMAL | 12,690,204 | 0.000% | 4s | TIME_LIMIT | 12,615,324 | 7.525% | 601s | 64 feas / 38 opt (102 iters) | Gurobi |
| F1_system_scale_M seed 1 | OPTIMAL | 19,418,604 | 0.071% | 33s | TIME_LIMIT | 18,842,844 | 22.306% | 601s | 139 feas / 7 opt (146 iters) | Gurobi |
| F2_mode_tradeoff_H seed 1 | OPTIMAL | 17,273,013 | 0.034% | 27s | TIME_LIMIT | 16,749,573 | 27.313% | 600s | 263 feas / 8 opt (271 iters) | Gurobi |
| F4_operational_conflict_H seed 1 | OPTIMAL | 18,942,129 | 0.000% | 22s | TIME_LIMIT | 18,443,649 | 23.625% | 600s | 168 feas / 8 opt (176 iters) | Gurobi |

Instances: 5. Solved to the gap target: Gurobi 5, Benders 0. Wins: Gurobi 5, Benders 0.
Total Benders cuts across all runs: 1010 feasibility, 64 optimality.
