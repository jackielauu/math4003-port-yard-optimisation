| Instance | Gurobi status | Gurobi obj | Gurobi gap | Gurobi time | Benders status | Benders obj | Benders gap | Benders time | Benders cuts | Faster / better |
|---|---|---|---|---|---|---|---|---|---|---|
| F1_system_scale_H seed 1 | OPTIMAL | 28,897,122 | 0.049% | 13s | TIME_LIMIT | 26,537,442 | 37.802% | 600s | 347 feas / 27 opt (374 iters) | Gurobi |
| F1_system_scale_L seed 1 | OPTIMAL | 12,690,204 | 0.000% | 1s | TIME_LIMIT | 12,629,484 | 10.674% | 600s | 64 feas / 22 opt (86 iters) | Gurobi |
| F1_system_scale_M seed 1 | OPTIMAL | 19,650,444 | 0.000% | 5s | TIME_LIMIT | 19,085,724 | 20.408% | 601s | 183 feas / 51 opt (234 iters) | Gurobi |
| F2_mode_tradeoff_H seed 1 | OPTIMAL | 17,505,573 | 0.027% | 4s | TIME_LIMIT | 16,837,653 | 26.884% | 600s | 233 feas / 47 opt (280 iters) | Gurobi |
| F4_operational_conflict_H seed 1 | OPTIMAL | 19,024,929 | 0.000% | 2s | TIME_LIMIT | 18,745,509 | 19.440% | 600s | 205 feas / 13 opt (218 iters) | Gurobi |

Instances: 5. Solved to the gap target: Gurobi 5, Benders 0. Wins: Gurobi 5, Benders 0.
Total Benders cuts across all runs: 1032 feasibility, 160 optimality.
