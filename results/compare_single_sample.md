| Instance | Gurobi status | Gurobi obj | Gurobi gap | Gurobi time | Benders status | Benders obj | Benders gap | Benders time | Faster / better |
|---|---|---|---|---|---|---|---|---|---|
| F1_system_scale_L seed 1 | OPTIMAL | 12,690,204 | 0.000% | 0s | TIME_LIMIT | 12,659,484 | 12.908% | 600s | Gurobi |
| F1_system_scale_M seed 1 | TIME_LIMIT | 19,585,164 | 0.279% | 600s | TIME_LIMIT | 18,561,564 | 30.156% | 600s | Gurobi (smaller gap) |
| F2_mode_tradeoff_H seed 1 | OPTIMAL | 17,494,773 | 0.089% | 44s | TIME_LIMIT | 16,793,253 | 34.893% | 600s | Gurobi |
| F2_mode_tradeoff_L seed 1 | OPTIMAL | 18,095,279 | 0.092% | 592s | TIME_LIMIT | 17,553,119 | 24.497% | 600s | Gurobi |
| F2_mode_tradeoff_M seed 1 | OPTIMAL | 17,698,955 | 0.097% | 105s | TIME_LIMIT | 17,084,075 | 27.225% | 600s | Gurobi |
| F3_yard_flow_H seed 1 | TIME_LIMIT | 20,265,280 | 1.028% | 600s | TIME_LIMIT | -5,343,307,881 | 13544.313% | 600s | Gurobi (smaller gap) |
| F3_yard_flow_L seed 1 | TIME_LIMIT | 19,193,351 | 0.187% | 600s | TIME_LIMIT | 18,549,431 | 29.164% | 600s | Gurobi (smaller gap) |
| F3_yard_flow_M seed 1 | TIME_LIMIT | 18,039,206 | 0.134% | 600s | TIME_LIMIT | -6,145,149,830 | 15561.830% | 600s | Gurobi (smaller gap) |
| F4_operational_conflict_H seed 1 | TIME_LIMIT | 18,977,559 | 0.221% | 600s | TIME_LIMIT | 18,089,379 | 30.949% | 600s | Gurobi (smaller gap) |
| F4_operational_conflict_L seed 1 | OPTIMAL | 17,550,170 | 0.097% | 76s | TIME_LIMIT | 16,859,690 | 37.710% | 600s | Gurobi |
| F4_operational_conflict_M seed 2 | TIME_LIMIT | 18,139,367 | 0.186% | 600s | TIME_LIMIT | 17,034,362 | 38.350% | 600s | Gurobi (smaller gap) |

Instances: 11. Solved to the gap target: Gurobi 5, Benders 0. Wins: Gurobi 11, Benders 0.
