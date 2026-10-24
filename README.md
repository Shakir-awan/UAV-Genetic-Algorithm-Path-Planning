# Multi-Objective UAV Fleet Path Planning with Genetic Algorithms

A compact, object-oriented Python implementation for the Artificial Intelligence Assignment 2 problem at Bahria University, Lahore Campus. Three UAV routes are optimized together on a 50 × 50 hazard grid. The fitness score rewards priority-weighted target detection and penalizes energy use, flight time, environmental exposure, no-fly segment crossings, and flight-range excess.

## Project Overview

The simulation contains 90 no-fly cells, four continuous Gaussian hazard pockets, and 20 disaster targets with priorities from 1.5 to 5.0. Each of the three UAVs has five real-valued waypoints, a speed in the range 12.0–28.0 m/s, and sensor power in the range 1.5–4.5. The sensor radius is `R = power × 2.0` grid units.

The source code includes a waypoint repair method based on eight-neighbour lookup, 12 intermediate samples per route segment, tournament selection, segment-swapping crossover, spatial Gaussian mutation, two-member elitism, and a 3,200-evaluation random-search baseline. It creates a best-versus-average convergence curve, an evolved trajectory/hazard map, and coverage snapshots from generations 1, 25, 50, and 75.

## System Architecture

| File / component | Responsibility |
|---|---|
| `uav_strategy_engine.py` | Complete simulation, objective function, GA operators, baseline, plots, and JSON summary |
| `GridEnvironment` | Grid, 90 blocked cells, Gaussian hazard map, 20 targets, and launch points |
| `UAVChromosome` | Float waypoint genes, speed and sensor power; clipping and neighbour-repair logic |
| `compute_fitness()` | Coverage reward minus energy, time, hazard, collision, and range costs |
| `custom_segment_crossover()` | Exchanges a contiguous waypoint section for one drone |
| `spatial_mutation()` | Gaussian coordinate mutation and operating-parameter mutation |
| `run_genetic_algorithm()` | Tournament selection, elitism, evaluation history, and stopping at generation 75 |
| `run_random_search()` | Baseline of 3,200 independently generated candidate plans |
| `technical_report.md` | Full technical report for the 3–5 page submission requirement |
| `requirements.txt` | Pinned Python library dependencies |

## Parameters

| Parameter | Value |
|---|---:|
| Grid size | 50 × 50 |
| No-fly obstacle cells | 90 |
| Continuous Gaussian hazard pockets | 4 |
| Disaster targets | 20 |
| Target priority range | 1.5–5.0 |
| Number of drones | 3 |
| Waypoints per drone | 5 |
| Speed range | 12.0–28.0 m/s |
| Sensor power range | 1.5–4.5 |
| Sensor radius | `R = P × 2.0` grid units |
| Population size | 40 |
| Generations | 75 |
| Mutation rate | 0.25 |
| Elitism | Top 2 |
| Tournament size | 3 |
| Intermediate samples per segment | 12 |
| Fitness weights | `w_cov=3.0`, `w_eng=0.08`, `w_time=0.15`, `w_haz=1.1` |
| Energy model | `(0.04 × v² + 1.8 × P) × t` |
| Random-search evaluations | 3,200 |

## Installation

Use Python 3.10 or later. From the repository folder, run:

```bash
python -m pip install -r requirements.txt
