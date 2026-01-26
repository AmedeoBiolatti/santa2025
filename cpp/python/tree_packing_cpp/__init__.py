"""
C++ Tree Packing Optimization Library

This module provides a high-performance C++ implementation of tree packing
optimization algorithms, including:

- ALNS (Adaptive Large Neighborhood Search)
- Simulated Annealing
- Various ruin and recreate operators

Example usage:
    import tree_packing_cpp as tp

    # Create problem
    problem = tp.Problem.create_tree_packing_problem()

    # Create initial solution
    solution = tp.Solution.init_random(67, side=10.0, seed=42)

    # Run optimization
    best_score, overall_score = tp.run_optimization(
        problem,
        solution,
        num_iterations=5000,
        seed=42,
        n_remove=1,
        max_recreate=1,
        initial_temp=1000.0,
        cooling_rate=0.9995,
        verbose=True
    )

    print(f"Best score: {best_score}")
"""

try:
    from .tree_packing_cpp import *
except ImportError:
    # Module not yet built
    import warnings
    warnings.warn(
        "tree_packing_cpp C++ module not found. "
        "Build with: cd cpp/build && cmake .. && make"
    )

__version__ = "1.0.0"
__all__ = [
    # Types
    "Vec2",
    "TreeParams",
    "TreeParamsSoA",
    "Solution",
    "SolutionEval",
    "Problem",
    "GlobalState",
    "RNG",
    # Optimizers
    "Optimizer",
    "OptimizerState",
    "RandomRuin",
    "CellRuin",
    "RandomRecreate",
    "NoiseOptimizer",
    "Chain",
    "ALNS",
    "SimulatedAnnealing",
    "CoolingSchedule",
    # Functions
    "run_optimization",
    # Constants
    "PI",
    "CENTER_Y",
    "CENTER_R",
    "THR",
]
