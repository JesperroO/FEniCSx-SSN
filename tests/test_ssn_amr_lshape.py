"""
Test: AMR-SSN on L-Shaped Domain with Reentrant Corner Singularity
==================================================================
Verification case for the adaptive mesh refinement engine.

Expected behaviour
------------------
- Initial coarse mesh (lc ≈ 0.08): ~400 cells, uniform
- Each AMR level: gradient-jump indicator concentrates near the
  reentrant corner at (0.5, 0.5) → Dörfler marking → local refinement
- After 4 AMR levels: cells near corner are ~16× smaller than initial,
  cells far from corner remain coarse
- SSN iteration count stays ≤ 10 at every AMR level (mesh independence)

Outputs
-------
  assets/amr_fig1_mesh_levels.png    — mesh at each AMR level
  assets/amr_fig2_indicator.png      — jump indicator field (final level)
  assets/amr_fig3_convergence.png    — SSN residuals per AMR level
  output_2d/amr_level_<k>_*.xdmf    — ParaView datasets per level
"""

import sys, os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpi4py import MPI
import dolfinx
from dolfinx import mesh as dmesh, fem, io
from dolfinx.fem.petsc import (
    assemble_matrix, assemble_vector,
    apply_lifting, set_bc, create_matrix, create_vector,
)
from petsc4py import PETSc
import ufl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ssn.amr_engine import compute_jump_indicator, dorfer_mark, amr_refine, transfer_scalar

try:
    import gmsh
    from dolfinx.io import gmshio
except ImportError:
    print("gmsh not found — run inside the dolfinx Docker container.")
    sys.exit(1)

os.makedirs("assets",    exist_ok=True)
os.makedirs("output_2d", exist_ok=True)

comm = MPI.COMM_WORLD
rank = comm.rank

# ── problem parameters ─────────────────────────────────────────────────────────
alpha = 0.05
nu    = 1e-3

# Target: Gaussian centred exactly at the reentrant corner
#   Forces state and adjoint to interact with the geometric singularity
def make_yd(domain):
    x = ufl.SpatialCoordinate(domain)
    return 8.0 * ufl.exp(-60.0 * ((x[0]-0.5)**2 + (x[1]-0.5)**2))

# AMR settings
MAX_AMR_LEVELS = 5
THETA          = 0.8    # Dörfler fraction
ETA_TOL        = 5e-3   # stop when max indicator < this


# ── L-shaped mesh ──────────────────────────────────────────────────────────────
def create_lshape_mesh(comm, lc=0.08):
    """
    L-shaped domain = unit square minus top-right quadrant.
    Vertices: (0,0)→(1,0)→(1,0.5)→(0.5,0.5)→(0.5,1)→(0,1)
    Reentrant corner at (0.5, 0.5) — interior angle 270°.
    """
    gmsh.initialize()
    if rank == 0:
        gmsh.model.add("L_shape_AMR")
        pts = [
            gmsh.model.geo.addPoint(0.0, 0.0, 0, lc),
            gmsh.model.geo.addPoint(1.0, 0.0, 0, lc),
            gmsh.model.geo.addPoint(1.0, 0.5, 0, lc),
            gmsh.model.geo.addPoint(0.5, 0.5, 0, lc),  # reentrant corner
            gmsh.model.geo.addPoint(0.5, 1.0, 0, lc),
            gmsh.model.geo.addPoint(0.0, 1.0, 0, lc),
        ]
        lines = [gmsh.model.geo.addLine(pts[i], pts[(i+1) % 6]) for i in range(6)]
        cl   = gmsh.model.geo.addCurveLoop(lines)
        surf = gmsh.model.geo.addPlaneSurface([cl])
        gmsh.model.geo.synchronize()
        gmsh.model.addPhysicalGroup(2, [surf], 1)
        gmsh.model.mesh.generate(2)

    domain, _, _ = gmshio.model_to_mesh(gmsh.model, comm, 0, gdim=2)
    gmsh.finalize()
    return domain


# ── FEM infrastructure builder ─────────────────────────────────────────────────
def build_fem(domain, alpha, nu):
    """
    Build mixed P1³ space, BCs, UFL forms, KSP on `domain`.
    Returns (W, bcs, w_k, forms_dict, ksp, vol_form).
    """
    P1 = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1)
    W  = fem.FunctionSpace(domain, ufl.MixedElement([P1, P1, P1]))

    fdim    = domain.topology.dim - 1
    domain.topology.create_entities(fdim)
    domain.topology.create_connectivity(fdim, domain.topology.dim)
    bfacets = dmesh.exterior_facet_indices(domain.topology)
    dofs_y  = fem.locate_dofs_topological(W.sub(0), fdim, bfacets)
    dofs_p  = fem.locate_dofs_topological(W.sub(2), fdim, bfacets)
    bcs     = [
        fem.dirichletbc(PETSc.ScalarType(0), dofs_y, W.sub(0)),
        fem.dirichletbc(PETSc.ScalarType(0), dofs_p, W.sub(2)),
    ]

    w_k = fem.Function(W)
    y, u, p = ufl.split(w_k)
    dy, du, dp = ufl.TrialFunctions(W)
    v_y, v_u, v_p = ufl.TestFunctions(W)

    y_d = make_yd(domain)

    chi_A_plus  = ufl.conditional(ufl.gt(p,  alpha), 1.0, 0.0)
    chi_A_minus = ufl.conditional(ufl.lt(p, -alpha), 1.0, 0.0)
    chi_A       = chi_A_plus + chi_A_minus
    u_expected  = chi_A_plus*(p-alpha)/nu + chi_A_minus*(p+alpha)/nu

    a = ( (ufl.inner(ufl.grad(dy), ufl.grad(v_y)) + dy*v_y - du*v_y) * ufl.dx
        + (ufl.inner(ufl.grad(dp), ufl.grad(v_p)) + dp*v_p + dy*v_p) * ufl.dx
        + (du*v_u - (chi_A/nu)*dp*v_u) * ufl.dx )

    L = ( -(ufl.inner(ufl.grad(y), ufl.grad(v_y)) + y*v_y - u*v_y) * ufl.dx
        - (ufl.inner(ufl.grad(p), ufl.grad(v_p)) + p*v_p + (y-y_d)*v_p) * ufl.dx
        - (u - u_expected) * v_u * ufl.dx )

    a_form   = fem.form(a)
    L_form   = fem.form(L)
    vol_form = fem.form(chi_A * ufl.dx)

    A_mat = create_matrix(a_form)
    b_vec = create_vector(L_form)
    dw    = fem.Function(W)

    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A_mat)
    ksp.setType(PETSc.KSP.Type.PREONLY)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.LU)
    pc.setFactorSolverType("mumps")

    forms = dict(a=a_form, L=L_form, A=A_mat, b=b_vec, dw=dw)
    return W, bcs, w_k, forms, ksp, vol_form


# ── SSN solve ─────────────────────────────────────────────────────────────────
def run_ssn(W, bcs, w_k, forms, ksp, vol_form, max_iter=30):
    A_mat, b_vec, dw = forms["A"], forms["b"], forms["dw"]
    a_form, L_form   = forms["a"], forms["L"]
    residuals  = []
    prev_vol   = -1.0
    conv_iter  = -1

    for i in range(max_iter):
        A_mat.zeroEntries()
        assemble_matrix(A_mat, a_form, bcs=bcs)
        A_mat.assemble()

        with b_vec.localForm() as loc:
            loc.set(0)
        assemble_vector(b_vec, L_form)
        apply_lifting(b_vec, [a_form], [bcs])
        b_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b_vec, bcs)

        res     = b_vec.norm()
        act_vol = comm.allreduce(fem.assemble_scalar(vol_form), op=MPI.SUM)
        residuals.append(res)

        if res < 1e-10 or (res < 1e-8 and abs(act_vol - prev_vol) < 1e-12):
            conv_iter = i
            break

        prev_vol = act_vol
        ksp.solve(b_vec, dw.vector)
        dw.x.scatter_forward()
        w_k.x.array[:] += dw.x.array[:]
        w_k.x.scatter_forward()

    return residuals, conv_iter


# ── XDMF export ───────────────────────────────────────────────────────────────
def export_level(domain, w_k, eta_vals, DG0, level):
    # Function.sub().collapse() -> Function  (dolfinx 0.7)
    y_out = w_k.sub(0).collapse(); y_out.name = "state_y"
    u_out = w_k.sub(1).collapse(); u_out.name = "control_u"
    p_out = w_k.sub(2).collapse(); p_out.name = "adjoint_p"

    eta_func = fem.Function(DG0, name="jump_indicator")
    n_local  = DG0.dofmap.index_map.size_local
    eta_func.x.array[:n_local] = eta_vals[:n_local]
    eta_func.x.scatter_forward()

    for tag, func in [("state", y_out), ("control", u_out),
                      ("adjoint", p_out), ("indicator", eta_func)]:
        path = f"output_2d/amr_level{level}_{tag}.xdmf"
        with io.XDMFFile(comm, path, "w") as xf:
            xf.write_mesh(domain)
            xf.write_function(func)
    if rank == 0:
        print(f"    XDMF: output_2d/amr_level{level}_*.xdmf")


# ── matplotlib helper ─────────────────────────────────────────────────────────
def get_triang(domain):
    coords = domain.geometry.x
    conn   = domain.topology.connectivity(2, 0)
    cells  = np.array([conn.links(i) for i in range(conn.num_nodes)])
    return mtri.Triangulation(coords[:, 0], coords[:, 1], cells)


# ── main AMR loop ─────────────────────────────────────────────────────────────
def run_amr_test():
    if rank == 0:
        print("=" * 65)
        print("  AMR-SSN: L-SHAPED DOMAIN WITH REENTRANT CORNER SINGULARITY")
        print("=" * 65)

    domain = create_lshape_mesh(comm, lc=0.08)

    # Per-level data for plots
    level_residuals  = []   # list of residual arrays
    level_cell_counts = []
    level_domains    = []
    level_etas       = []

    w_prev = None   # transferred solution from previous level

    for level in range(MAX_AMR_LEVELS):
        n_cells = domain.topology.index_map(2).size_global
        n_dofs  = 0  # computed after build

        if rank == 0:
            print(f"\n── AMR level {level}  ({n_cells} cells) " + "─" * 30)

        W, bcs, w_k, forms, ksp, vol_form = build_fem(domain, alpha, nu)
        n_dofs = W.dofmap.index_map.size_global * W.dofmap.index_map_bs

        if rank == 0:
            print(f"   DOFs: {n_dofs}")

        # Warm-start with transferred solution from previous level
        if w_prev is not None:
            domain_old = w_prev["domain"]
            # FunctionSpace.sub().collapse() -> (SubSpace, dof_map_list)
            for sub_idx, arr_key in [(0, "y"), (1, "u"), (2, "p")]:
                _old        = w_prev[arr_key]               # Function on domain_old
                V_sub, dmap = W.sub(sub_idx).collapse()     # sub-space + index map
                dmap        = np.asarray(dmap, dtype=np.int32)
                _new        = transfer_scalar(_old, domain_old, domain)
                n_local     = V_sub.dofmap.index_map.size_local
                w_k.x.array[dmap[:n_local]] = _new.x.array[:n_local]
            w_k.x.scatter_forward()
            if rank == 0:
                print("   Warm-started from previous level.")

        # Run SSN
        t0 = time.time()
        residuals, conv_iter = run_ssn(W, bcs, w_k, forms, ksp, vol_form)
        elapsed = time.time() - t0

        if rank == 0:
            status = f"converged in {conv_iter} steps" if conv_iter >= 0 else "MAX ITER"
            print(f"   SSN: {status}  |  wall time: {elapsed:.1f}s")
            for k, r in enumerate(residuals):
                mark = " ←" if k == conv_iter else ""
                print(f"     iter {k}: {r:.4e}{mark}")

        level_residuals.append(residuals)
        level_cell_counts.append(n_cells)

        # Extract adjoint for error estimation
        _p = w_k.sub(2).collapse()
        p_func = _p[0] if isinstance(_p, tuple) else _p

        eta_vals, DG0 = compute_jump_indicator(p_func, domain)
        max_eta = comm.allreduce(float(eta_vals.max()), op=MPI.MAX)
        sum_eta = comm.allreduce(float(eta_vals.sum()), op=MPI.SUM)

        if rank == 0:
            print(f"   Jump indicator: max={max_eta:.4e}, sum={sum_eta:.4e}")

        # Store mesh snapshot and eta for plotting (rank 0 only)
        if rank == 0:
            level_domains.append(domain)
            level_etas.append(eta_vals.copy())

        export_level(domain, w_k, eta_vals, DG0, level)

        # Clean up PETSc objects before potential refine
        ksp.destroy()
        forms["A"].destroy()
        forms["b"].destroy()

        if max_eta < ETA_TOL:
            if rank == 0:
                print(f"\n  Indicator below tolerance ({ETA_TOL:.1e}). AMR done.")
            break

        if level == MAX_AMR_LEVELS - 1:
            if rank == 0:
                print("  Max AMR levels reached.")
            break

        # Dörfler marking
        marked = dorfer_mark(eta_vals, theta=THETA)
        if rank == 0:
            pct = 100 * len(marked) / max(len(eta_vals), 1)
            print(f"   Dörfler: marking {len(marked)} / {len(eta_vals)} cells ({pct:.1f}%)")

        # Store solution components for next level warm-start
        # Function.sub().collapse() -> Function  (dolfinx 0.7)
        w_prev = {
            "domain": domain,
            "y": w_k.sub(0).collapse(),
            "u": w_k.sub(1).collapse(),
            "p": w_k.sub(2).collapse(),
        }

        # Refine
        domain_new, _ = amr_refine(domain, marked)
        domain = domain_new

    # ── visualizations (rank 0) ───────────────────────────────────────────────
    if rank == 0:
        _render_mesh_levels(level_domains, level_etas, level_cell_counts)
        _render_indicator(level_domains, level_etas)
        _render_convergence(level_residuals, level_cell_counts)

        print("\n" + "=" * 65)
        print("  OUTPUT")
        print("=" * 65)
        print("  assets/amr_fig1_mesh_levels.png")
        print("  assets/amr_fig2_indicator.png")
        print("  assets/amr_fig3_convergence.png")
        print("  output_2d/amr_level<k>_{state,control,adjoint,indicator}.xdmf")
        print("=" * 65)


# ── plot helpers ──────────────────────────────────────────────────────────────
def _render_mesh_levels(domains, etas, cell_counts):
    n = len(domains)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5), facecolor="white")
    if n == 1:
        axes = [axes]

    for lv, (ax, domain, eta, nc) in enumerate(zip(axes, domains, etas, cell_counts)):
        triang = get_triang(domain)
        coords = domain.geometry.x

        # Colour by log(h_K) — cell size indicator
        conn = domain.topology.connectivity(2, 0)
        h_cell = np.zeros(conn.num_nodes)
        for ci in range(conn.num_nodes):
            verts = conn.links(ci)
            pts   = coords[verts, :2]
            # Approximate h as max edge length
            diffs = [np.linalg.norm(pts[i]-pts[j]) for i in range(3) for j in range(i+1,3)]
            h_cell[ci] = max(diffs)

        tc = ax.tripcolor(triang, h_cell, cmap="YlOrRd_r", shading="flat")
        ax.triplot(triang, color="gray", lw=0.2, alpha=0.4)
        # Mark reentrant corner
        ax.plot(0.5, 0.5, "r*", markersize=12, zorder=10, label="Reentrant corner")
        ax.set_title(f"Level {lv}\n{nc} → {conn.num_nodes} cells",
                     fontsize=11, fontweight="bold")
        ax.set_aspect("equal")
        ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
        fig.colorbar(tc, ax=ax, label="$h_K$", shrink=0.8)

    plt.suptitle("AMR Mesh Evolution — Cell Size $h_K$\n"
                 "(dark = fine near corner, light = coarse far away)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("assets/amr_fig1_mesh_levels.png", dpi=180, bbox_inches="tight")
    plt.close()
    print("  ✓  assets/amr_fig1_mesh_levels.png")


def _render_indicator(domains, etas):
    if not domains:
        return
    # Show first (coarsest) and last (finest) indicator side by side
    pairs = [(0, "Initial (coarse)"), (len(domains)-1, "Final (refined)")]
    fig, axes = plt.subplots(1, len(pairs), figsize=(7*len(pairs), 5.5),
                             facecolor="white")
    if len(pairs) == 1:
        axes = [axes]

    for ax, (idx, label) in zip(axes, pairs):
        domain  = domains[idx]
        eta     = etas[idx]
        triang  = get_triang(domain)
        conn    = domain.topology.connectivity(2, 0)
        cell_x  = np.array([domain.geometry.x[conn.links(i), 0].mean() for i in range(conn.num_nodes)])
        cell_y  = np.array([domain.geometry.x[conn.links(i), 1].mean() for i in range(conn.num_nodes)])

        n_local = min(len(eta), conn.num_nodes)
        eta_plot = np.zeros(conn.num_nodes)
        eta_plot[:n_local] = eta[:n_local]

        tc = ax.tripcolor(triang, eta_plot, cmap="hot_r", shading="flat")
        ax.triplot(triang, color="white", lw=0.2, alpha=0.3)
        ax.plot(0.5, 0.5, "b*", markersize=14, zorder=10, label="Reentrant corner")
        ax.set_title(f"{label}\nJump indicator $\\eta_K = h_K\\,|\\![\\![\\nabla p \\cdot n]\\!]\\!|^2$",
                     fontsize=11, fontweight="bold")
        ax.set_aspect("equal")
        ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
        ax.legend(fontsize=9)
        fig.colorbar(tc, ax=ax, label="$\\eta_K$", shrink=0.85)

    plt.suptitle("Gradient-Jump Error Indicator $\\eta_K$ (adjoint field $p$)\n"
                 "Indicator concentrates at reentrant corner → drives local refinement",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("assets/amr_fig2_indicator.png", dpi=180, bbox_inches="tight")
    plt.close()
    print("  ✓  assets/amr_fig2_indicator.png")


def _render_convergence(level_residuals, cell_counts):
    n = len(level_residuals)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor="white")

    # Left: SSN residuals per AMR level
    ax = axes[0]
    cmap = plt.cm.plasma(np.linspace(0.1, 0.9, n))
    for lv, (res, nc) in enumerate(zip(level_residuals, cell_counts)):
        ax.semilogy(range(len(res)), res, "o-", color=cmap[lv], lw=2,
                    markersize=7, label=f"Level {lv}  ({nc} cells)")
    ax.set_title("SSN Residual per AMR Level\n(mesh independence: step count stable)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Newton Iteration"); ax.set_ylabel(r"$\|F\|_{L^2}$ (log)")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, which="both", ls="--", alpha=0.4)

    # Right: DOF count vs final residual
    ax2 = axes[1]
    final_res = [res[-1] for res in level_residuals]
    ax2.semilogy(cell_counts, final_res, "D-", color="#e74c3c", lw=2, markersize=10)
    for lv, (nc, fr) in enumerate(zip(cell_counts, final_res)):
        ax2.annotate(f"L{lv}", (nc, fr), textcoords="offset points",
                     xytext=(6, 4), fontsize=10)
    ax2.set_title("Final SSN Residual vs. Mesh Size\n(AMR: same accuracy, fewer cells)",
                  fontsize=12, fontweight="bold")
    ax2.set_xlabel("Cell count at AMR level start")
    ax2.set_ylabel("Final residual $\\|F\\|_{L^2}$")
    ax2.grid(True, which="both", ls="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig("assets/amr_fig3_convergence.png", dpi=180, bbox_inches="tight")
    plt.close()
    print("  ✓  assets/amr_fig3_convergence.png")


if __name__ == "__main__":
    run_amr_test()
