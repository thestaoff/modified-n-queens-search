# modified_nqueens.py
# CSE307 / AI — Assignment 1: Modified N-Queens (BFS, UCS, A*)
# Based on the movement and goal rules defined for the assignment.

from __future__ import annotations
from dataclasses import dataclass
from collections import deque
import heapq
from typing import Dict, List, Optional, Set, Tuple, Iterable

Pos = Tuple[int, int]                 # (row, col)
State = Tuple[Pos, ...]               # length N, one position per queen index 0..N-1
Action = Tuple[int, str]              # (queen_index, 'U'/'D'/'L'/'R')


DIRS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}


@dataclass(frozen=True)
class SearchResult:
    actions: List[Action]
    cost: int
    nodes_expanded: int
    frontier_remaining: int
    final_state: State


class ModifiedNQueensProblem:
    """
    State: tuple of N (row,col) positions — queen i is at state[i]
    Initial: all queens in top row, one per column: ((0,0),(0,1),...,(0,N-1))
    Actions restriction: a queen cannot move in a direction if the adjacent cell
                        is occupied by another queen.
    Result restriction: if move goes off-board OR adjacent cell occupied => disallowed,
                        and the state remains unchanged (per spec).
    Goal:
      1) every row contains exactly one queen
      2) no conflicts: same row, same column, or same diagonal
    Heuristic (for A*): total number of conflicts counted for every queen
                        (so each conflicting pair counts twice).
    """

    def __init__(self, n: int):
        if n < 1:
            raise ValueError("N must be >= 1")
        self.n = n
        self.initial: State = tuple((0, c) for c in range(n))

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.n and 0 <= c < self.n

    def actions(self, state: State) -> List[Action]:
        occupied = set(state)
        acts: List[Action] = []
        for i, (r, c) in enumerate(state):
            for d, (dr, dc) in DIRS.items():
                nr, nc = r + dr, c + dc
                # "Blocked moves in the Actions Function":
                # cannot move if adjacent cell is occupied
                if (nr, nc) in occupied:
                    continue
                # We DO allow proposing moves that go off-board here;
                # Result() will convert them into no-ops (state unchanged),
                # matching the spec.
                acts.append((i, d))
        return acts

    def result(self, state: State, action: Action) -> State:
        i, d = action
        r, c = state[i]
        dr, dc = DIRS[d]
        nr, nc = r + dr, c + dc

        occupied = set(state)

        # "Blocked moves in the Result Function":
        # if off-board OR adjacent occupied => disallowed => unchanged
        if not self.in_bounds(nr, nc):
            return state
        if (nr, nc) in occupied:
            return state

        new_positions = list(state)
        new_positions[i] = (nr, nc)
        return tuple(new_positions)

    def step_cost(self, state: State, action: Action, next_state: State) -> int:
        # Each attempted move costs 1 (standard move cost).
        return 1

    # --- Goal / Conflicts ---

    @staticmethod
    def _conflict(p: Pos, q: Pos) -> bool:
        (r1, c1), (r2, c2) = p, q
        if r1 == r2:
            return True
        if c1 == c2:
            return True
        if abs(r1 - r2) == abs(c1 - c2):
            return True
        return False

    def is_goal(self, state: State) -> bool:
        # Condition 1: every row contains exactly one queen
        rows = [r for r, _ in state]
        if len(set(rows)) != self.n:
            return False

        # Condition 2: no conflicts
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self._conflict(state[i], state[j]):
                    return False
        return True

    def heuristic(self, state: State) -> int:
        # Total conflicts counted for every queen (double-count pairs).
        conflicts = 0
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    continue
                if self._conflict(state[i], state[j]):
                    conflicts += 1
        return conflicts


# ---------------- Search Algorithms ----------------

def _reconstruct_path(parent: Dict[State, Tuple[Optional[State], Optional[Action]]],
                      goal: State) -> List[Action]:
    actions: List[Action] = []
    cur = goal
    while True:
        prev, act = parent[cur]
        if prev is None or act is None:
            break
        actions.append(act)
        cur = prev
    actions.reverse()
    return actions


def bfs(problem: ModifiedNQueensProblem) -> Optional[SearchResult]:
    start = problem.initial
    if problem.is_goal(start):
        return SearchResult([], 0, 0, 1, start)

    q = deque([start])
    in_frontier: Set[State] = {start}
    explored: Set[State] = set()
    parent: Dict[State, Tuple[Optional[State], Optional[Action]]] = {start: (None, None)}
    nodes_expanded = 0

    while q:
        s = q.popleft()
        in_frontier.discard(s)
        explored.add(s)
        nodes_expanded += 1

        for a in problem.actions(s):
            ns = problem.result(s, a)
            # Avoid no-op successors causing infinite loops
            if ns == s:
                continue
            if ns in explored or ns in in_frontier:
                continue

            parent[ns] = (s, a)

            if problem.is_goal(ns):
                acts = _reconstruct_path(parent, ns)
                return SearchResult(
                    actions=acts,
                    cost=len(acts),
                    nodes_expanded=nodes_expanded,
                    frontier_remaining=len(q) + 1,  # +1 for ns that would be enqueued
                    final_state=ns,
                )

            q.append(ns)
            in_frontier.add(ns)

    return None


def ucs(problem: ModifiedNQueensProblem) -> Optional[SearchResult]:
    start = problem.initial
    if problem.is_goal(start):
        return SearchResult([], 0, 0, 1, start)

    # heap items: (g, tie, state)
    heap: List[Tuple[int, int, State]] = []
    tie = 0
    heapq.heappush(heap, (0, tie, start))

    best_g: Dict[State, int] = {start: 0}
    parent: Dict[State, Tuple[Optional[State], Optional[Action]]] = {start: (None, None)}
    closed: Set[State] = set()
    nodes_expanded = 0

    while heap:
        g, _, s = heapq.heappop(heap)
        if s in closed:
            continue
        closed.add(s)
        nodes_expanded += 1

        if problem.is_goal(s):
            acts = _reconstruct_path(parent, s)
            return SearchResult(
                actions=acts,
                cost=g,
                nodes_expanded=nodes_expanded,
                frontier_remaining=len(heap),
                final_state=s,
            )

        for a in problem.actions(s):
            ns = problem.result(s, a)
            if ns == s:
                continue
            ng = g + problem.step_cost(s, a, ns)
            if ns in closed and ng >= best_g.get(ns, float("inf")):
                continue
            if ng < best_g.get(ns, float("inf")):
                best_g[ns] = ng
                parent[ns] = (s, a)
                tie += 1
                heapq.heappush(heap, (ng, tie, ns))

    return None


def astar(problem: ModifiedNQueensProblem) -> Optional[SearchResult]:
    start = problem.initial
    if problem.is_goal(start):
        return SearchResult([], 0, 0, 1, start)

    # heap items: (f, g, tie, state)
    heap: List[Tuple[int, int, int, State]] = []
    tie = 0
    h0 = problem.heuristic(start)
    heapq.heappush(heap, (h0, 0, tie, start))

    best_g: Dict[State, int] = {start: 0}
    parent: Dict[State, Tuple[Optional[State], Optional[Action]]] = {start: (None, None)}
    closed: Set[State] = set()
    nodes_expanded = 0

    while heap:
        f, g, _, s = heapq.heappop(heap)
        if s in closed:
            continue
        closed.add(s)
        nodes_expanded += 1

        if problem.is_goal(s):
            acts = _reconstruct_path(parent, s)
            return SearchResult(
                actions=acts,
                cost=g,
                nodes_expanded=nodes_expanded,
                frontier_remaining=len(heap),
                final_state=s,
            )

        for a in problem.actions(s):
            ns = problem.result(s, a)
            if ns == s:
                continue
            ng = g + problem.step_cost(s, a, ns)
            if ns in closed and ng >= best_g.get(ns, float("inf")):
                continue
            if ng < best_g.get(ns, float("inf")):
                best_g[ns] = ng
                parent[ns] = (s, a)
                tie += 1
                hn = problem.heuristic(ns)
                heapq.heappush(heap, (ng + hn, ng, tie, ns))

    return None


# ---------------- Convenience Runner ----------------

def solve_and_print(n: int) -> None:
    prob = ModifiedNQueensProblem(n)

    for name, fn in [("BFS", bfs), ("UCS", ucs), ("A*", astar)]:
        res = fn(prob)
        if res is None:
            print(f"{name} (N={n}): No solution found.")
            continue
        print(f"{name} (N={n}):")
        print(f"  cost = {res.cost}")
        print(f"  actions = {res.actions}")
        print(f"  nodes expanded = {res.nodes_expanded}")
        print(f"  frontier remaining = {res.frontier_remaining}")
        print(f"  final state = {res.final_state}")
        print()


if __name__ == "__main__":
    # Example: run for N = 4..8
    for N in range(4, 9):
        solve_and_print(N)
