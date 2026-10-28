| Instance | Gurobi status | Gurobi obj | Gurobi gap | Gurobi time | Benders status | Benders obj | Benders gap | Benders time | Benders cuts | Faster / better |
|---|---|---|---|---|---|---|---|---|---|---|
| F1_system_scale_H seed 1 | OPTIMAL | 28,867,122 | 0.000% | 6s | TIME_LIMIT | 28,126,242 | 19.692% | 600s | 92 feas / 41 opt (133 iters) | Gurobi |
| F1_system_scale_L seed 1 | OPTIMAL | 12,690,204 | 0.000% | 0s | TIME_LIMIT | 12,609,804 | 12.400% | 600s | 0 feas / 27 opt (27 iters) | Gurobi |
| F1_system_scale_M seed 1 | OPTIMAL | 19,640,844 | 0.049% | 2s | TIME_LIMIT | 18,981,804 | 22.301% | 600s | 10 feas / 12 opt (22 iters) | Gurobi |
| F2_mode_tradeoff_H seed 1 | OPTIMAL | 17,502,693 | 0.044% | 4s | TIME_LIMIT | 16,970,853 | 25.159% | 600s | 33 feas / 35 opt (68 iters) | Gurobi |
| F4_operational_conflict_H seed 1 | OPTIMAL | 19,013,409 | 0.061% | 1s | TIME_LIMIT | 18,583,352 | 21.001% | 600s | 26 feas / 17 opt (43 iters) | Gurobi |

Instances: 5. Solved to the gap target: Gurobi 5, Benders 0. Wins: Gurobi 5, Benders 0.
Total Benders cuts across all runs: 161 feasibility, 132 optimality.
