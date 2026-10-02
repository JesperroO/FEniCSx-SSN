"""
AMR Engine for SSN Solver
=========================
Four functions that compose the adaptive mesh refinement loop:

  compute_jump_indicator  — gradient-jump error estimator (DG0, interior facets)
  dorfer_mark             — Dörfler bulk marking (theta fraction of total error)
  amr_refine              — local refinement wrapper around dolfinx.mesh.refine
  transfer_scalar         — P1 function transfer between parent/child meshes
"""

import numpy as np
from dolfinx import fem, mesh as dmesh
from dolfinx.fem.petsc import (
    assemble_matrix, assemble_vector, create_matrix, create_vector,
)
from petsc4py import PETSc
import ufl


def compute_jump_indicator(p_func, domain):
    """
    Per-cell error indicator based on the gradient jump of the adjoint field p.

    For each cell K:
        eta_K = h_K * ||[[grad(p)·n]]||^2_{L2(∂K ∩ Ω)}

    where [[·]] denotes the jump across interior facets and h_K is the cell
    diameter.  The indicator is assembled into a DG0 function space so that
    each degree of freedom corresponds to exactly one cell.

    Returns
    -------
    eta_vals : np.ndarray, shape (n_cells_local,)
        Non-negative per-cell indicator values.
    DG0 : FunctionSpace
        The DG0 space used (needed to build mesh tags for marking).
    """
    DG0 = fem.FunctionSpace(domain, ("DG", 0))
    eta = fem.Function(DG0)

    n   = ufl.FacetNormal(domain)
    h   = ufl.CellDiameter(domain)
    phi = ufl.TestFunction(DG0)
    psi = ufl.TrialFunction(DG0)

    # Scalar normal-derivative jump:  [[grad(p)·n]] = grad(p)(+)·n(+) + grad(p)(-)·n(-)
    jump_pn = ufl.jump(ufl.grad(p_func), n)

    # Accumulate h * jump² over interior facets, split equally to both neighbours
    L_form = fem.form(
        h("+") * jump_pn**2 * (phi("+") + phi("-")) * 0.5 * ufl.dS
    )
    # Diagonal DG0 mass matrix (one DOF per cell)
    a_form = fem.form(ufl.inner(psi, phi) * ufl.dx)

    A = create_matrix(a_form)
    b = create_vector(L_form)

    assemble_matrix(A, a_form)
    A.assemble()
    assemble_vector(b, L_form)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    # Jacobi (diagonal) solve — exact for DG0 mass matrix
    ksp = PETSc.KSP().create(domain.comm)
    ksp.setOperators(A)
    ksp.setType(PETSc.KSP.Type.PREONLY)
    ksp.getPC().setType(PETSc.PC.Type.JACOBI)
    ksp.solve(b, eta.vector)
    eta.x.scatter_forward()
    ksp.destroy()
    A.destroy()
    b.destroy()

    # Guard against numerical noise
    eta_vals = np.maximum(eta.x.array.copy(), 0.0)
    return eta_vals, DG0


def dorfer_mark(eta_vals, theta=0.8):
    """
    Dörfler (bulk) marking strategy.

    Selects the fewest cells whose combined indicator value covers at least
    `theta` of the total.  This ensures the marked set is minimal yet
    captures the dominant error sources.

    Parameters
    ----------
    eta_vals : np.ndarray
        Per-cell indicator values (non-negative).
    theta : float
        Bulk fraction threshold, typically 0.7–0.9.

    Returns
    -------
    marked : np.ndarray[int32]
        Indices of cells to be refined, sorted by decreasing indicator.
    """
    if len(eta_vals) == 0:
        return np.array([], dtype=np.int32)

    total = eta_vals.sum()
    if total < 1e-30:
        return np.array([], dtype=np.int32)

    order   = np.argsort(eta_vals)[::-1]           # descending
    cumsum  = np.cumsum(eta_vals[order])
    cutoff  = np.searchsorted(cumsum, theta * total)
    cutoff  = min(cutoff + 1, len(eta_vals))        # inclusive

    return order[:cutoff].astype(np.int32)


def amr_refine(domain, marked_cells):
    """
    Refine `domain` at the cells listed in `marked_cells`.

    dolfinx 0.7 refines by *edge* index, not cell index.  This function
    converts cell → edge via cell-to-edge connectivity, then calls
    dolfinx.mesh.refine.

    Parameters
    ----------
    domain : dolfinx.mesh.Mesh
    marked_cells : np.ndarray[int32]

    Returns
    -------
    new_domain : dolfinx.mesh.Mesh
    """
    marked_cells = np.asarray(marked_cells, dtype=np.int32)

    # Build cell → edge (dim 2 → dim 1) connectivity
    domain.topology.create_connectivity(domain.topology.dim, 1)
    c2e = domain.topology.connectivity(domain.topology.dim, 1)

    edge_set = set()
    for ci in marked_cells:
        edge_set.update(c2e.links(ci).tolist())
    edge_indices = np.array(sorted(edge_set), dtype=np.int32)

    # dolfinx 0.7: refine(mesh, edges, redistribute) -> Mesh
    new_domain = dmesh.refine(domain, edge_indices, redistribute=False)

    # Rebuild required connectivity tables on the new mesh
    new_domain.topology.create_connectivity(
        new_domain.topology.dim - 1, new_domain.topology.dim
    )
    return new_domain, None


def transfer_scalar(u_old, domain_old, domain_new):
    """
    Transfer a scalar P1 Function from `domain_old` to `domain_new` by
    evaluating `u_old` at every DOF coordinate of the new function space.

    This approach is mesh-agnostic: it works for any pair of geometrically
    overlapping meshes, not just parent/child pairs.  The bounding-box tree
    handles points that fall exactly on cell boundaries robustly.

    Parameters
    ----------
    u_old     : fem.Function (Lagrange, degree 1) on domain_old
    domain_old : dolfinx.mesh.Mesh
    domain_new : dolfinx.mesh.Mesh

    Returns
    -------
    u_new : fem.Function on domain_new with the same name and element.
    """
    from dolfinx.geometry import (
        bb_tree,
        compute_collisions_points,
        compute_colliding_cells,
    )

    V_new = fem.FunctionSpace(domain_new, u_old.function_space.ufl_element())
    u_new = fem.Function(V_new)
    u_new.name = u_old.name

    n_local      = V_new.dofmap.index_map.size_local
    all_coords   = V_new.tabulate_dof_coordinates()           # (n_dofs, 3)
    local_coords = all_coords[:n_local, :domain_old.geometry.dim]

    # Geometric search on the old mesh
    bb         = bb_tree(domain_old, domain_old.topology.dim)
    candidates = compute_collisions_points(bb, local_coords)
    collisions = compute_colliding_cells(domain_old, candidates, local_coords)

    # Batch evaluation at valid points
    valid_pts   = []
    valid_cells = []
    valid_dofs  = []

    for i in range(n_local):
        cell_list = collisions.links(i)
        if len(cell_list) > 0:
            valid_pts.append(local_coords[i])
            valid_cells.append(cell_list[0])
            valid_dofs.append(i)

    if valid_pts:
        pts_3d = np.zeros((len(valid_pts), 3), dtype=np.float64)
        pts_3d[:, :domain_old.geometry.dim] = np.array(valid_pts)
        cells_arr = np.array(valid_cells, dtype=np.int32)
        vals = u_old.eval(pts_3d, cells_arr)      # shape (n, value_size)
        for j, dof_idx in enumerate(valid_dofs):
            u_new.x.array[dof_idx] = vals[j, 0]

    u_new.x.scatter_forward()
    return u_new
