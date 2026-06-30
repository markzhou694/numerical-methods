try:
    from .newton import newton_solve
    from .jacobi import jacobi_solve
    from .gauss_seidel import gauss_seidel_solve
    from .projection_2d import projection_2d_solve
    from .greedy_projection import greedy_projection_solve
    from .trimres import trimres
except ImportError:
    from newton import newton_solve
    from jacobi import jacobi_solve
    from gauss_seidel import gauss_seidel_solve
    from projection_2d import projection_2d_solve
    from greedy_projection import greedy_projection_solve
    from trimres import trimres
