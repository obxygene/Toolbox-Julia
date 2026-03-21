"""
graphene_qhe.jl  –  Reproduce Garcia et al. PRB 91, 245140 (2015) Fig. 2

Quantized Hall conductance of disordered graphene in a perpendicular
magnetic field, computed via the Kubo-Bastin KPM formula.

Run with multi-threading:
  julia --threads=4 graphene_qhe.jl

Parameter guide for quantized σ_xy plateaux:
  Fast test:  Lx=64,  Ly=256,  M=1024, R=10  (~minutes)
  Good run:   Lx=128, Ly=512,  M=1024, R=20  (~15 min)   [default below]
  Paper:      Lx=128, Ly=1024, M=6144, R=40  (~hours)
"""

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using KernelPolynomialMethod
using SparseArrays
using LinearAlgebra
using Printf

# -------------------------------------------------------
# Physical parameters
# -------------------------------------------------------
const t     = 2.8          # NN hopping (eV)
const gamma = 0.1 * t      # Anderson disorder strength

# System size — phi/phi0 = 1/Lx ensures Lx*(phi/phi0)=integer (PBC gauge)
const Lx = 128
const Ly = 512
const phi_over_phi0 = 1 / Lx   # flux per rectangular unit cell

const N_sites = 2 * Lx * Ly

@printf("Graphene QHE: Lx=%d, Ly=%d, N=%d sites\n", Lx, Ly, N_sites)
@printf("phi/phi0 = 1/%d = %.2e\n", Lx, phi_over_phi0)
@printf("Disorder gamma = %.2f*t\n", gamma/t)

# -------------------------------------------------------
# Build Hamiltonian and current operators
# -------------------------------------------------------
# Rectangular honeycomb unit cell: a1=(1,0), a2=(0,sqrt(3)), 2 atoms/cell
# Landau gauge A = (-By, 0, 0):
#   Peierls phase: -(π*phi/phi0/A_cell) * (yA+yB) * dx

const a2_y  = sqrt(3)
const d_AB  = (0.0, 1/sqrt(3))
const A_cell = sqrt(3)
const peierls_factor = π * phi_over_phi0 / A_cell

# NN bond displacements from A(ix,iy): cell shift and displacement
nn_cell_shift = ((0,0), (-1,0), (0,-1))
nn_disp       = (d_AB, (-1 + d_AB[1], d_AB[2]), (d_AB[1], -a2_y + d_AB[2]))

# Site index: (ix, iy, sublattice) -> 1-based global index
function site_idx(ix, iy, s)
    return (mod(iy-1, Ly)*Lx + mod(ix-1, Lx)) * 2 + s
end

println("Building Hamiltonian...")
t0 = time()

max_nnz = 8 * N_sites
h_I  = Vector{Int}(undef, max_nnz)
h_J  = Vector{Int}(undef, max_nnz)
h_V  = Vector{ComplexF64}(undef, max_nnz)
jx_I = Vector{Int}(undef, max_nnz); jx_J = Vector{Int}(undef, max_nnz)
jx_V = Vector{ComplexF64}(undef, max_nnz)
jy_I = Vector{Int}(undef, max_nnz); jy_J = Vector{Int}(undef, max_nnz)
jy_V = Vector{ComplexF64}(undef, max_nnz)
cnt  = 0

for iy in 1:Ly, ix in 1:Lx
    global cnt
    iA = site_idx(ix, iy, 1)
    rA = (Float64(ix-1), Float64(iy-1) * a2_y)

    for k in 1:3
        ds   = nn_cell_shift[k]
        iB   = site_idx(ix + ds[1], iy + ds[2], 2)
        dr   = nn_disp[k]
        rBx  = rA[1] + dr[1]
        rBy  = rA[2] + dr[2]

        phase_AB = -peierls_factor * (rA[2] + rBy) * dr[1]
        hop      = -t * exp(im * phase_AB)

        cnt += 1
        h_I[cnt]=iA; h_J[cnt]=iB; h_V[cnt]=hop
        jx_I[cnt]=iA; jx_J[cnt]=iB; jx_V[cnt]=im*hop*dr[1]
        jy_I[cnt]=iA; jy_J[cnt]=iB; jy_V[cnt]=im*hop*dr[2]

        cnt += 1
        h_I[cnt]=iB; h_J[cnt]=iA; h_V[cnt]=conj(hop)
        jx_I[cnt]=iB; jx_J[cnt]=iA; jx_V[cnt]=im*conj(hop)*(-dr[1])
        jy_I[cnt]=iB; jy_J[cnt]=iA; jy_V[cnt]=im*conj(hop)*(-dr[2])
    end
end

H_clean = sparse(h_I[1:cnt], h_J[1:cnt], h_V[1:cnt], N_sites, N_sites)
Jx      = sparse(jx_I[1:cnt], jx_J[1:cnt], jx_V[1:cnt], N_sites, N_sites)
Jy      = sparse(jy_I[1:cnt], jy_J[1:cnt], jy_V[1:cnt], N_sites, N_sites)

disorder = gamma .* (rand(N_sites) .- 0.5)
H_sparse = H_clean + spdiagm(0 => complex.(disorder))

@printf("Build time: %.1f s\n", time()-t0)

# -------------------------------------------------------
# KPM Convergence Study
# -------------------------------------------------------
M_list  = [512, 1024, 2048]
SR_list = [20,  20,   20  ]

E_range      = 0.5 * t
E_fermi_list = range(-E_range, E_range, length=300)
N_E          = length(E_fermi_list)

# Scale Hamiltonian once
epsilon_kpm = 0.05
H_tilde, a_scale, b_scale = kpm_scale_hamiltonian(H_sparse; epsilon=epsilon_kpm)

# Honeycomb prefactor: 4/a² × (2 atoms/cell) / A_cell = 4/a² × 2/√3
N_cells   = N_sites ÷ 2
prefactor = 4.0 / (a_scale^2) * (N_sites / N_cells) / A_cell

@printf("Prefactor = %.4f (e^2/h)\n", prefactor)

E_F_sc = clamp.((collect(E_fermi_list) .- b_scale) ./ a_scale,
                 -1 + epsilon_kpm/2, 1 - epsilon_kpm/2)

results = []

for (run_idx, (M, SR)) in enumerate(zip(M_list, SR_list))
    @printf("\nRun %d/%d: M=%d, R=%d\n", run_idx, length(M_list), M, SR)

    use_par = Threads.nthreads() > 1

    t1 = time()
    Mu_xy = kpm_moments_correlator(H_tilde, SR, M, Jx, Jy; parallel=use_par)
    @printf("  xy moments: %.1f s\n", time()-t1)

    t1 = time()
    Mu_xx = kpm_moments_correlator(H_tilde, SR, M, Jx, Jx; parallel=use_par)
    @printf("  xx moments: %.1f s\n", time()-t1)

    Mu_xy_K = kpm_kernel_correction(Mu_xy, M; kernel="Jackson")
    Mu_xx_K = kpm_kernel_correction(Mu_xx, M; kernel="Jackson")

    sig_xy = zeros(N_E)
    sig_xx = zeros(N_E)
    m_vec  = 0:(M-1)

    for (ii, EF) in enumerate(E_F_sc)
        Γ            = kpm_correlator_basis(EF, M)
        sig_xy[ii]   = prefactor * real(sum(Mu_xy_K .* Γ))
        T_EF         = cos.(m_vec .* acos(EF))
        sig_xx[ii]   = prefactor * real(dot(T_EF, Mu_xx_K * T_EF))
    end

    push!(results, (M=M, SR=SR, sigma_xy=sig_xy, sigma_xx=sig_xx))
    println("  Conductivities done.")
end

# -------------------------------------------------------
# Summary
# -------------------------------------------------------
println("\n=== Summary ===")
@printf("phi/phi0 = 1/%d,  E1 ~ %.3f t,  gamma = %.2f t\n",
        Lx, 3*sqrt(π*phi_over_phi0), gamma/t)
println("Expected: sigma_xy plateaux at ±2, ±6, ±10 e²/h")
println("Expected: sigma_xx peaks at Landau levels")
println("Results stored in `results` array (M, SR, sigma_xy, sigma_xx).")

# -------------------------------------------------------
# Optional: save results for plotting
# -------------------------------------------------------
# using DelimitedFiles
# for r in results
#     fname = "sigma_M$(r.M)_R$(r.SR).dat"
#     writedlm(fname, [collect(E_fermi_list) r.sigma_xx r.sigma_xy])
#     @printf("Saved %s\n", fname)
# end
