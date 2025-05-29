# Evacuation Transport Optimization for Assisted Residents during Nuclear Disasters

Source Code for the Paper:  
Development of an Evacuation Transport Model for Residents in Need of Assistance in Evacuation during Nuclear Disasters”

This repository contains the source code used in the above research paper.

## Overview

This project builds a Capacitated Vehicle Routing Problem (CVRP) model for transporting assisted residents (e.g., elderly, disabled) under nuclear disaster scenarios.  
Geographic data and elevation thresholds are used to construct feasible road networks for evacuation.

### Features

- Uses OpenStreetMap and official elevation data
- Implements two optimization methods:
  1. Exact optimization using the Gurobi Solver
  2. Heuristic optimization using a Genetic Algorithm (GA)

## Gurobi Optimization

### Target Directory  
`20250511_elevationXXm_gurubi_client12m/`  
(Replace `XX` with the elevation threshold in meters. For example: `elevation6m` excludes roads below 6 meters.)

### Steps

1. Run the script `CVRP_Geography_v7.py` in the `Geography/` folder.

   This script filters roads below the specified elevation and generates a road network and travel-time matrix.

   Example: include areas at or above 6 meters

   ```python
   elev = 6
   geo.get_filtered_road_network(output_file=f"omaezaki_≤{elev}melev.graphml", elev=elev, nrate=0.5)
   print('Network recreated', time.time() - start)
   ```

2. Run the script `CVRP_gurobi_3d_v2.py` in the `Optimization/` folder.

   This script optimizes the evacuation plan using the Gurobi solver based on the generated data.

## Genetic Algorithm Optimization

### Target Directory  
`20250417_elevation0m_npop500_ngen30000/`  
(GA is only applied to the case with elevation = 0m)

### Steps

1. Run the script `CVRP_Geography_v7.py` in the `Geography/` folder.

2. Run the script `CVRP_Calculation_3d_v2.py` in the `Optimization/` folder.

   This script uses a Genetic Algorithm to search for optimized evacuation routes.

## Notes

- Geographic data is based on OpenStreetMap and elevation sources.
- The optimization model is formulated as a Capacitated Vehicle Routing Problem (CVRP).
- To reproduce results, a Python environment and Gurobi installation are required.
  A valid Gurobi license is necessary.
