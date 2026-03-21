# validate_conductance_honeycomb.jl
# Small-system validation: KPM σ_xy vs exact diagonalisation for a honeycomb
# lattice in a perpendicular magnetic field (symmetric Peierls gauge).
#
# Lattice: a1=(1,0), a2=(0,√3), A_cell=√3
# Peierls: symmetric gauge, phase on A→B bond = −π(Φ/Φ₀)/A_cell · (yA+yB)·Δx
#
# Reference: Validate_KPM_Conductance_Honeycomb.m

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using KernelPolynomialMethod
using SparseArrays
using LinearAlgebra
using Plots

# ===== Parameters =====
const t   = 1.0
Lx  = 128         # unit cells in x
Ly  = 128         # unit cells in y
phi = 1/Lx        # flux per unit cell (Φ/Φ₀)

N_sites = 2 * Lx * Ly

println("=== Validate KPM vs Exact  (Honeycomb, Φ/Φ₀ = 1/$Lx) ===")
println("$(Lx)×$(Ly) cells, N = $N_sites sites")

# ===== Lattice geometry =====
a1    = [1.0, 0.0]
a2    = [0.0, sqrt(3)]
d_AB  = [0.0, 1/sqrt(3)]
A_cell = sqrt(3)
peierls_fac = π * phi / A_cell

nn_disp       = (d_AB, -a1 .+ d_AB, -a2 .+ d_AB)
nn_cell_shift = ([0,0], [-1,0], [0,-1])

idx = (ix, iy, s) -> (mod(iy-1,Ly)*Lx + mod(ix-1,Lx))*2 + s

# ===== Build H, Jx, Jy (NN only, no NNN) =====
println("Building H, Jx, Jy ...")

h_rows  = Int[]; h_cols  = Int[]; h_vals  = ComplexF64[]
jx_rows = Int[]; jx_cols = Int[]; jx_vals = ComplexF64[]
jy_rows = Int[]; jy_cols = Int[]; jy_vals = ComplexF64[]

for iy in 1:Ly, ix in 1:Lx
    iA = idx(ix, iy, 1)
    rA = Float64[(ix-1)*a1[1], (iy-1)*a2[2]]

    for k in 1:3
        ds = nn_cell_shift[k]
        iB = idx(ix + ds[1], iy + ds[2], 2)
        dr = nn_disp[k]
        rB = rA .+ dr

        # Symmetric-gauge Peierls phase on A→B bond
        phase_AB = -peierls_fac * (rA[2] + rB[2]) * dr[1]
        hop = -t * exp(1im * phase_AB)

        push!(h_rows, iA); push!(h_cols, iB); push!(h_vals, hop)
        push!(h_rows, iB); push!(h_cols, iA); push!(h_vals, conj(hop))
        push!(jx_rows, iA); push!(jx_cols, iB); push!(jx_vals,  1im*hop*dr[1])
        push!(jx_rows, iB); push!(jx_cols, iA); push!(jx_vals,  1im*conj(hop)*(-dr[1]))
        push!(jy_rows, iA); push!(jy_cols, iB); push!(jy_vals,  1im*hop*dr[2])
        push!(jy_rows, iB); push!(jy_cols, iA); push!(jy_vals,  1im*conj(hop)*(-dr[2]))
    end
end

H_sp  = sparse(h_rows,  h_cols,  h_vals,  N_sites, N_sites)
Jx_sp = sparse(jx_rows, jx_cols, jx_vals, N_sites, N_sites)
Jy_sp = sparse(jy_rows, jy_cols, jy_vals, N_sites, N_sites)

println("H  Hermitian check: $(round(norm(H_sp - H_sp', Inf)/norm(H_sp, Inf); sigdigits=2))")
println("Jx Hermitian check: $(round(norm(Jx_sp - Jx_sp', Inf)/max(1,norm(Jx_sp,Inf)); sigdigits=2))")

# ===== Exact diagonalisation =====
println("\nDiagonalising ($N_sites sites)...")
t_ed = @elapsed begin
    F = eigen(Hermitian(Matrix(H_sp)))
    eigvals = F.values
    V = F.vectors
end
println("Done in $(round(t_ed; digits=1)) s.  Bandwidth: [$(round(eigvals[1];digits=3)), $(round(eigvals[end];digits=3))]")

# ===== Exact σ_xy =====
# σ_xy/(e²/ℏ) = (1/Ω) Σ_{n<EF,m>EF} 2 Im[⟨n|Jx|m⟩⟨m|Jy|n⟩] / (Em−En)²
# Convert to e²/h: × 2π
Ω = (N_sites / 2) * A_cell   # system area

Jx_eig = V' * Jx_sp * V
Jy_eig = V' * Jy_sp * V

E_fermi = range(-2.5t, 2.5t; length=200) |> collect
sigma_xy_exact = zeros(length(E_fermi))

println("Computing exact σ_xy...")
t_exact = @elapsed begin
    for (ii, EF) in enumerate(E_fermi)
        occ   = findall(eigvals .< EF)
        unocc = findall(eigvals .> EF)
        S = 0.0
        for n in occ, m in unocc
            dE = eigvals[m] - eigvals[n]
            abs(dE) > 1e-10 || continue
            S += 2imag(Jx_eig[n,m] * Jy_eig[m,n]) / dE^2
        end
        sigma_xy_exact[ii] = S / Ω   # units of e²/ℏ
    end
end
println("Done in $(round(t_exact; digits=1)) s")

# Convert to e²/h
sigma_xy_exact_h = sigma_xy_exact .* 2π

# ===== KPM σ_xy =====
println("\nComputing KPM σ_xy, σ_xx ...")
t_kpm = @elapsed begin
    sigma_xy_kpm, sigma_xx_kpm = kpm_hall_conductance(H_sp, Jx_sp, Jy_sp, E_fermi;
        N_moments = 256,
        N_randvec = 10,
        kernel    = "Jackson",
        epsilon   = 0.05)
end
# kpm_hall_conductance uses 4/a² for 1-atom/cell.
# Honeycomb: 2 atoms/cell, A_cell=√3  →  multiply by 2/√3
honeycomb_correction = 2 / sqrt(3)
sigma_xy_kpm .*= honeycomb_correction
sigma_xx_kpm .*= honeycomb_correction
println("Done in $(round(t_kpm; digits=1)) s")

# ===== KPM σ_xx via T_m·T_n formula (Kubo-Greenwood) =====
println("Computing σ_xx with T_m·T_n formula...")
H_t, a_sc, b_sc = kpm_scale_hamiltonian(H_sp; epsilon=0.05)
Mu_xx = kpm_moments_correlator(H_t, 10, 256, Jx_sp, Jx_sp)
Mu_xx_K = kpm_kernel_correction(Mu_xx, 256; kernel="Jackson")

m_arr = 0:255
sigma_xx_TmTn = map(E_fermi) do EF
    EF_sc = clamp((EF - b_sc)/a_sc, -1+0.025, 1-0.025)
    T_vec = cos.(m_arr .* acos(EF_sc))
    real((4/a_sc^2) * honeycomb_correction * (T_vec' * Mu_xx_K * T_vec))
end

# ===== Ratio check =====
signif = findall(abs.(sigma_xy_exact_h) .> 0.5)
if !isempty(signif)
    ratio = sigma_xy_kpm[signif] ./ sigma_xy_exact_h[signif]
    valid = isfinite.(ratio) .& (abs.(ratio) .< 100)
    r = ratio[valid]
    println("\n=== Honeycomb: σ_xy ratio KPM/Exact ===")
    println("Mean  : $(round(mean(r); digits=4))")
    println("Median: $(round(median(r); digits=4))")
    println("Std   : $(round(std(r); digits=4))  (target ≈ 1.0)")
end

# ===== Plot =====
# σ_xy comparison
p1 = plot_conductance_compare(E_fermi, sigma_xy_kpm, sigma_xy_exact_h,
                               sigma_xx_kpm, sigma_xx_TmTn;
                               plateaux = [-10,-6,-2,2,6,10],
                               title = "Honeycomb $(Lx)×$(Ly), Φ/Φ₀=1/$Lx: KPM vs Exact")
display(p1)
savefig(p1, joinpath(@__DIR__, "validate_honeycomb.png"))
println("Saved validate_honeycomb.png")
