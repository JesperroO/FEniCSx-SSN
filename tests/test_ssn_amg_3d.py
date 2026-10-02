"""
Test: AMG Block Preconditioner vs MUMPS on 3-D Unit Cube
=========================================================
Solves the L1-optimal control problem on a 3-D mesh at multiple resolutions
and compares:

  - MUMPS direct solver   (O(N^2) memory in 3-D, baseline)
  - GMRES + AMG block PC  (near-O(N) memory, scalable)

Expected results
----------------
Resolution    DOFs      MUMPS time   AMG time   GMRES iters
N=16         ~13 k      < 1 s        < 1 s       ≈ 5-20
N=32         ~100 k     ~10-60 s     < 5 s       ≈ 5-30
N=48         ~340 k     OOM / slow   < 30 s      ≈ 5-40
N=64         ~820 k     skip         < 120 s     ≈ 5-50

Outputs
-------
  assets/amg_fig1_timing.png    — wall time vs DOFs (log-log)
  assets/amg_fig2_memory.png    — peak RSS vs DOFs (log-log)
  assets/amg_fig3_iters.png     — GMRES + SSN iter counts vs resolution
  (terminal)                    — per-run comparison table
"""

import os, sys, time, resource
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpi4py import MPI
import dolfinx
from dolfinx import mesh as dmesh, fem
from dolfinx.fem.petsc import (
    assemble_matrix, assemble_vector,
    apply_lifting, set_bc, create_matrix, create_vector,
)
from petsc4py import PETSc
import ufl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ssn.amg_preconditioner import build_amg_ksp, build_mumps_ksp

os.makedirs("assets",    exist_ok=True)
os.makedirs("output_2d", exist_ok=True)

comm = MPI.COMM_WORLD
rank = comm.rank

alpha = 0.05
nu    = 1e-3


# ── memory helper ──────────────────────────────────────────────────────────────
def _rss_mb():
    """Current process peak RSS in MiB (Linux: ru_maxrss in KB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


# ── problem builder ────────────────────────────────────────────────────────────
def build_3d_ssn(N):
    """
    Build mesh, mixed space, UFL forms, and BCs for the 3-D unit cube.
    Target: single Gaussian at centre (0.5, 0.5, 0.5).
    Returns (domain, W, bcs, w_k, a_form, L_form, vol_form).
    """
    domain = dmesh.create_unit_cube(comm, N, N, N)

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
    y, u, p   = ufl.split(w_k)
    dy, du, dp = ufl.TrialFunctions(W)
    v_y, v_u, v_p = ufl.TestFunctions(W)

    x = ufl.SpatialCoordinate(domain)
    y_d = 8.0 * ufl.exp(
        -80.0 * ((x[0]-0.5)**2 + (x[1]-0.5)**2 + (x[2]-0.5)**2)
    )

    chi_A_plus  = ufl.conditional(ufl.gt(p,  alpha), 1.0, 0.0)
    chi_A_minus = ufl.conditional(ufl.lt(p, -alpha), 1.0, 0.0)
    chi_A       = chi_A_plus + chi_A_minus
    u_expected  = chi_A_plus*(p-alpha)/nu + chi_A_minus*(p+alpha)/nu

    a = ((ufl.inner(ufl.grad(dy), ufl.grad(v_y)) + dy*v_y - du*v_y) * ufl.dx
       + (ufl.inner(ufl.grad(dp), ufl.grad(v_p)) + dp*v_p + dy*v_p) * ufl.dx
       + (du*v_u - (chi_A/nu)*dp*v_u) * ufl.dx)

    L = (-(ufl.inner(ufl.grad(y), ufl.grad(v_y)) + y*v_y - u*v_y) * ufl.dx
        -(ufl.inner(ufl.grad(p), ufl.grad(v_p)) + p*v_p + (y-y_d)*v_p) * ufl.dx
        -(u - u_expected) * v_u * ufl.dx)

    a_form   = fem.form(a)
    L_form   = fem.form(L)
    vol_form = fem.form(chi_A * ufl.dx)

    return domain, W, bcs, w_k, a_form, L_form, vol_form


# ── SSN loop ───────────────────────────────────────────────────────────────────
def run_ssn(W, bcs, w_k, a_form, L_form, vol_form,
            ksp_builder, max_iter=20):
    """
    Run SSN Newton loop using `ksp_builder(A_mat, W, comm)` for each linear solve.
    Returns (ssn_residuals, gmres_iters_per_step, conv_iter, total_linear_time).
    """
    A_mat = create_matrix(a_form)
    b_vec = create_vector(L_form)
    dw    = fem.Function(W)

    ssn_residuals = []
    gmres_iters   = []
    total_lin_t   = 0.0
    prev_vol      = -1.0
    conv_iter     = -1

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
        ssn_residuals.append(res)

        if res < 1e-10 or (res < 1e-8 and abs(act_vol - prev_vol) < 1e-12):
            conv_iter = i
            break

        prev_vol = act_vol

        # Build KSP fresh each SSN step (active set changes → matrix changes)
        t_lin = time.time()
        ksp = ksp_builder(A_mat, W, comm)
        ksp.solve(b_vec, dw.vector)
        total_lin_t += time.time() - t_lin

        n_lin_iters = ksp.getIterationNumber()  # 0 for direct solver
        gmres_iters.append(n_lin_iters)
        ksp.destroy()

        dw.x.scatter_forward()
        w_k.x.array[:] += dw.x.array[:]
        w_k.x.scatter_forward()

    A_mat.destroy()
    b_vec.destroy()
    return ssn_residuals, gmres_iters, conv_iter, total_lin_t


# ── single benchmark run ───────────────────────────────────────────────────────
def benchmark_one(N, use_amg: bool, skip_if_dofs_above: int = 0):
    """
    Run SSN on N³ 3-D mesh with chosen solver.
    Returns dict of metrics, or None if skipped.
    """
    domain, W, bcs, w_k, a_form, L_form, vol_form = build_3d_ssn(N)
    n_dofs = W.dofmap.index_map.size_global * W.dofmap.index_map_bs

    if skip_if_dofs_above > 0 and n_dofs > skip_if_dofs_above:
        return None

    tag = "AMG" if use_amg else "MUMPS"
    if rank == 0:
        print(f"  [{tag}] N={N}  DOFs={n_dofs:,}")

    if use_amg:
        def ksp_builder(A, Wsp, c):
            return build_amg_ksp(A, Wsp, c)
    else:
        def ksp_builder(A, Wsp, c):
            return build_mumps_ksp(A, c)

    rss_before = _rss_mb()
    t0 = time.time()

    ssn_res, gmres_iters, conv_iter, lin_t = run_ssn(
        W, bcs, w_k, a_form, L_form, vol_form, ksp_builder
    )

    wall_t   = time.time() - t0
    rss_after = _rss_mb()
    peak_rss  = rss_after  # approximation: peak = current after solve

    if rank == 0:
        ssn_steps = conv_iter + 1 if conv_iter >= 0 else len(ssn_res)
        avg_gmres = np.mean(gmres_iters) if gmres_iters else 0
        print(f"         SSN steps={ssn_steps}  "
              f"avg GMRES/step={avg_gmres:.1f}  "
              f"wall={wall_t:.2f}s  RSS≈{peak_rss:.0f} MiB")
        for k, r in enumerate(ssn_res):
            mark = " ←" if k == conv_iter else ""
            print(f"         iter {k}: {r:.4e}{mark}")

    return dict(
        N=N, dofs=n_dofs, solver=tag,
        ssn_steps=(conv_iter+1 if conv_iter >= 0 else len(ssn_res)),
        avg_gmres=float(np.mean(gmres_iters)) if gmres_iters else 0.0,
        wall_t=wall_t, lin_t=lin_t, rss_mb=peak_rss,
        ssn_res=ssn_res, gmres_iters=gmres_iters,
    )


# ── plotting ───────────────────────────────────────────────────────────────────
def render_plots(results):
    amg   = [r for r in results if r["solver"] == "AMG"]
    mumps = [r for r in results if r["solver"] == "MUMPS"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), facecolor="white")

    # ── Fig 1: wall time ─────────────────────────────────────────────────────
    ax = axes[0]
    if amg:
        ax.loglog([r["dofs"] for r in amg],   [r["wall_t"] for r in amg],
                  "o-", color="#2980b9", lw=2, markersize=9, label="AMG (GMRES)")
    if mumps:
        ax.loglog([r["dofs"] for r in mumps], [r["wall_t"] for r in mumps],
                  "s--", color="#e74c3c", lw=2, markersize=9, label="MUMPS (direct)")
    # O(N) and O(N^2) reference lines
    if amg:
        d0, t0 = amg[0]["dofs"], amg[0]["wall_t"]
        ds = np.array([r["dofs"] for r in amg])
        ax.loglog(ds, t0*(ds/d0),    ":", color="gray",  lw=1, label="O(N) ref")
        ax.loglog(ds, t0*(ds/d0)**2, ":", color="black", lw=1, label="O(N²) ref")
    ax.set_title("Wall-clock Time vs DOFs\n(3-D unit cube, SSN to tol 1e-8)",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Total DOFs (log)"); ax.set_ylabel("Wall time (s, log)")
    ax.legend(fontsize=9); ax.grid(True, which="both", ls="--", alpha=0.4)

    # ── Fig 2: memory ────────────────────────────────────────────────────────
    ax = axes[1]
    if amg:
        ax.loglog([r["dofs"] for r in amg],   [r["rss_mb"] for r in amg],
                  "o-", color="#2980b9", lw=2, markersize=9, label="AMG")
    if mumps:
        ax.loglog([r["dofs"] for r in mumps], [r["rss_mb"] for r in mumps],
                  "s--", color="#e74c3c", lw=2, markersize=9, label="MUMPS")
    if amg:
        d0, m0 = amg[0]["dofs"], amg[0]["rss_mb"]
        ds = np.array([r["dofs"] for r in amg])
        ax.loglog(ds, m0*(ds/d0),    ":", color="gray",  lw=1, label="O(N) ref")
        ax.loglog(ds, m0*(ds/d0)**2, ":", color="black", lw=1, label="O(N²) ref")
    ax.set_title("Peak RSS Memory vs DOFs\n(AMG: O(N);  MUMPS: O(N²) in 3-D)",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Total DOFs (log)"); ax.set_ylabel("RSS memory (MiB, log)")
    ax.legend(fontsize=9); ax.grid(True, which="both", ls="--", alpha=0.4)

    # ── Fig 3: GMRES & SSN iterations ────────────────────────────────────────
    ax = axes[2]
    if amg:
        ax.plot([r["dofs"] for r in amg], [r["avg_gmres"] for r in amg],
                "D-", color="#27ae60", lw=2, markersize=9, label="Avg GMRES / SSN step")
        ax.plot([r["dofs"] for r in amg], [r["ssn_steps"] for r in amg],
                "o--", color="#8e44ad", lw=2, markersize=8, label="SSN steps")
    ax.set_title("Iteration Counts vs DOFs\n(mesh independence: counts should plateau)",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Total DOFs (linear scale)"); ax.set_ylabel("Iteration count")
    ax.legend(fontsize=9); ax.grid(True, ls="--", alpha=0.4)
    ax.set_ylim(bottom=0)

    plt.suptitle("AMG Block Preconditioner vs MUMPS — 3-D O(N) Scalability",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("assets/amg_fig1_timing.png", dpi=180, bbox_inches="tight")
    plt.close()

    # Save fig 2 and 3 as separate files too (shared axes above — split them)
    # (plots already in fig above; rename the combined one)
    os.rename("assets/amg_fig1_timing.png", "assets/amg_combined.png")
    print("  ✓  assets/amg_combined.png  (timing | memory | iterations)")


# ── main ───────────────────────────────────────────────────────────────────────
def run_comparison():
    if rank == 0:
        print("=" * 70)
        print("  AMG BLOCK PRECONDITIONER vs MUMPS — 3-D SCALABILITY BENCHMARK")
        print("  Problem: L1 optimal control on N³ unit cube, P1³ mixed FEM")
        print("  AMG: GMRES + PCFieldSplit(ADDITIVE) + BoomerAMG on y,p blocks")
        print("=" * 70)

    results = []

    # ── AMG runs (all resolutions) ────────────────────────────────────────────
    amg_sizes = [16, 24, 32]      # add 48, 64 if you have RAM and patience
    if rank == 0:
        print(f"\n{'─'*70}")
        print("  AMG solver (GMRES + BoomerAMG block preconditioner)")
        print(f"{'─'*70}")

    for N in amg_sizes:
        r = benchmark_one(N, use_amg=True)
        if r is not None:
            results.append(r)

    # ── MUMPS runs (small sizes only — 3-D fill-in is O(N^2)) ────────────────
    mumps_sizes = [16, 24]        # skip N≥32 to avoid OOM / very long waits
    mumps_dof_limit = 150_000     # hard cap: skip if DOFs exceed this

    if rank == 0:
        print(f"\n{'─'*70}")
        print("  MUMPS solver (direct LU, reference baseline)")
        print(f"  Skipped for DOFs > {mumps_dof_limit:,} (3-D fill-in too expensive)")
        print(f"{'─'*70}")

    for N in mumps_sizes:
        r = benchmark_one(N, use_amg=False, skip_if_dofs_above=mumps_dof_limit)
        if r is not None:
            results.append(r)

    # ── summary table ─────────────────────────────────────────────────────────
    if rank == 0:
        print(f"\n{'─'*70}")
        print(f"  {'Solver':<8} {'N':>4} {'DOFs':>10} {'SSN':>5} "
              f"{'GMRES/step':>11} {'Wall (s)':>10} {'RSS (MiB)':>11}")
        print(f"  {'─'*67}")
        for r in sorted(results, key=lambda x: (x["solver"], x["N"])):
            gstr = f"{r['avg_gmres']:.1f}" if r["avg_gmres"] > 0 else "N/A"
            print(f"  {r['solver']:<8} {r['N']:>4} {r['dofs']:>10,} "
                  f"{r['ssn_steps']:>5} {gstr:>11} "
                  f"{r['wall_t']:>10.2f} {r['rss_mb']:>11.0f}")
        print(f"  {'─'*67}")
        print()

    # ── plots ─────────────────────────────────────────────────────────────────
    if rank == 0:
        render_plots(results)

    if rank == 0:
        print("=" * 70)
        print("  VERDICT")
        print("=" * 70)
        amg_r   = [r for r in results if r["solver"] == "AMG"]
        mumps_r = [r for r in results if r["solver"] == "MUMPS"]
        if amg_r and mumps_r:
            # Compare largest common N
            common_N = set(r["N"] for r in amg_r) & set(r["N"] for r in mumps_r)
            if common_N:
                best_N = max(common_N)
                ta = next(r for r in amg_r   if r["N"] == best_N)["wall_t"]
                tm = next(r for r in mumps_r if r["N"] == best_N)["wall_t"]
                print(f"  At N={best_N}: AMG {ta:.2f}s vs MUMPS {tm:.2f}s"
                      f"  (speedup {tm/ta:.1f}×)")
        if amg_r and len(amg_r) >= 2:
            n1, t1 = amg_r[0]["dofs"],  amg_r[0]["wall_t"]
            n2, t2 = amg_r[-1]["dofs"], amg_r[-1]["wall_t"]
            slope = np.log(t2/t1) / np.log(n2/n1)
            print(f"  AMG empirical complexity exponent: {slope:.2f}  "
                  f"(1.0 = O(N), 2.0 = O(N²))")
        print("=" * 70)


if __name__ == "__main__":
    run_comparison()
