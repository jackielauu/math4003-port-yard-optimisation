| Instance | Gurobi status | Gurobi obj | Gurobi gap | Gurobi time | Benders status | Benders obj | Benders gap | Benders time | Benders cuts | Faster / better |
|---|---|---|---|---|---|---|---|---|---|---|
| F1_system_scale_L seed 1 | OPTIMAL | 12,690,204 | 0.000% | 0s | TIME_LIMIT | 12,468,684 | 12.322% | 120s | 60 feas / 20 opt (80 iters) | Gurobi |
| F3_yard_flow_M seed 2 | OPTIMAL | -987,615,259 | -0.000% | 3s | TIME_LIMIT | -13,382,879,655 | 33772.704% | 120s | 272 feas / 1 opt (273 iters) | Gurobi |
| F4_operational_conflict_M seed 1 | OPTIMAL | 18,150,099 | 0.066% | 2s | TIME_LIMIT | 17,816,300 | 27.819% | 120s | 195 feas / 23 opt (218 iters) | Gurobi |

Instances: 3. Solved to the gap target: Gurobi 3, Benders 0. Wins: Gurobi 3, Benders 0.
Total Benders cuts across all runs: 527 feasibility, 44 optimality.
