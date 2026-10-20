# Modified N-Queens Search

A Python implementation of a movement-based N-Queens problem solved with uninformed and informed graph-search algorithms.

## Implemented algorithms

- Breadth-First Search (BFS)
- Uniform-Cost Search (UCS)
- A* search

## Problem model

- All queens begin in the top row, one per column.
- A move shifts one queen by one cell up, down, left, or right.
- Moves that leave the board or enter an occupied cell are rejected.
- A goal state contains one queen per row with no row, column, or diagonal conflicts.
- The A* heuristic counts queen conflicts.

## Run

The project uses only the Python standard library.

```bash
python modified_nqueens.py
```

The included runner evaluates board sizes 4 through 8 and prints the solution cost, action sequence, expanded-node count, remaining frontier size, and final state for each algorithm.

## Complexity note

The state space grows rapidly with `N`. BFS and UCS can require substantial time and memory on larger boards; use smaller values while experimenting with the model or heuristic.
