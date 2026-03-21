# benchmark_hall_haldane.jl
# KPM Hall conductance for the Haldane model on a honeycomb lattice.
# Chern insulator: σ_xy = ±e²/h in bulk gap when |M| < 3√3 t₂ sin(ϕ_H).
#
# Lattice vectors: a1=(1,0), a2=(1/2,√3/2)
# Unit cell: A and B sublattices, 2 atoms/cell
# NNN phase +ϕ_H on A sublattice, −ϕ_H on B sublattice.
#
# Reference: Haldane, PRL 61, 2015 (1988)

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using KernelPolynomialMethod
using SparseArrays
using LinearAlgebra
using Plots

# ===== Parameters =====
Lx    = 256       # unit cells in x
Ly    = 256       # unit cells in y
t1    = 1.0       # NN hopping
t2    = 0.1       # NNN hopping
ϕ_H   = π/2       # Haldane flux (maximises gap)
M_sub = 0.5       # sublattice mass

N_sites = 2 * Lx * Ly

println("Haldane: $(Lx)×$(Ly) cells, N = $N_sites sites")
println("t1=$(t1), t2=$(t2), ϕ_H=π/2, M=$(M_sub)")
println("Critical mass: 3√3·t₂·sin(ϕ_H) = $(round(3sqrt(3)*t2*sin(ϕ_H); digits=4))")

# KPM parameters
N_moments = 256
N_randvec = 5

# ===== Index helpers =====
idx = (ix, iy, s) -> (mod(iy-1, Ly)*Lx + mod(ix-1, Lx)) * 2 + s   # s ∈ {1=A, 2=B}

# ===== Lattice geometry =====
a1 = [1.0, 0.0]
a2 = [0.5, sqrt(3)/2]
d_sub = [0.0, 1/sqrt(3)]   # sublattice offset A→B within same cell

nn_disp = (d_sub, -a1 .+ d_sub, -a2 .+ d_sub)    # A→B bond vectors
nnn_disp = (a1, a2, a2 .- a1)                      # NNN bond vectors (same sublattice)

# ===== Build H, Jx, Jy =====
println("Building H, Jx, Jy ...")

h_rows  = Int[]; h_cols  = Int[]; h_vals  = ComplexF64[]
jx_rows = Int[]; jx_cols = Int[]; jx_vals = ComplexF64[]
jy_rows = Int[]; jy_cols = Int[]; jy_vals = ComplexF64[]

function add_bond!(iR, iC, hop, dr)
    push!(h_rows, iR); push!(h_cols, iC); push!(h_vals, hop)
    push!(h_rows, iC); push!(h_cols, iR); push!(h_vals, conj(hop))
    push!(jx_rows, iR); push!(jx_cols, iC); push!(jx_vals,  1im*hop*dr[1])
    push!(jx_rows, iC); push!(jx_cols, iR); push!(jx_vals,  1im*conj(hop)*(-dr[1]))
    push!(jy_rows, iR); push!(jy_cols, iC); push!(jy_vals,  1im*hop*dr[2])
    push!(jy_rows, iC); push!(jy_cols, iR); push!(jy_vals,  1im*conj(hop)*(-dr[2]))
end

for iy in 1:Ly, ix in 1:Lx
    iA = idx(ix, iy, 1)
    iB = idx(ix, iy, 2)

    # On-site sublattice mass (no current contribution)
    push!(h_rows, iA); push!(h_cols, iA); push!(h_vals, +M_sub)
    push!(h_rows, iB); push!(h_cols, iB); push!(h_vals, -M_sub)
    for v in (jx_vals, jy_vals)
        push!(v, 0.0im); push!(v, 0.0im)
    end
    for v in (jx_rows, jy_rows); push!(v, iA); push!(v, iB); end
    for v in (jx_cols, jy_cols); push!(v, iA); push!(v, iB); end

    # NN hoppings: A(ix,iy) → B in 3 directions
    nn_targets = (idx(ix, iy, 2),
                  idx(ix-1, iy, 2),
                  idx(ix, iy-1, 2))
    for k in 1:3
        add_bond!(iA, nn_targets[k], -t1, nn_disp[k])
    end

    # NNN hoppings: A sublattice (+ϕ_H)
    nnn_A = (idx(ix+1, iy, 1),
             idx(ix, iy+1, 1),
             idx(ix-1, iy+1, 1))
    for k in 1:3
        add_bond!(iA, nnn_A[k], -t2*exp(+1im*ϕ_H), nnn_disp[k])
    end

    # NNN hoppings: B sublattice (−ϕ_H)
    nnn_B = (idx(ix+1, iy, 2),
             idx(ix, iy+1, 2),
             idx(ix-1, iy+1, 2))
    for k in 1:3
        add_bond!(iB, nnn_B[k], -t2*exp(-1im*ϕ_H), nnn_disp[k])
    end
end

H  = sparse(h_rows,  h_cols,  h_vals,  N_sites, N_sites)
Jx = sparse(jx_rows, jx_cols, jx_vals, N_sites, N_sites)
Jy = sparse(jy_rows, jy_cols, jy_vals, N_sites, N_sites)

@assert norm(H  - H',  Inf) < 1e-10  "H not Hermitian"
@assert norm(Jx - Jx', Inf)/max(1, norm(Jx, Inf)) < 1e-10  "Jx not Hermitian"
@assert norm(Jy - Jy', Inf)/max(1, norm(Jy, Inf)) < 1e-10  "Jy not Hermitian"
println("Hermitian checks passed.")

# ===== DOS =====
println("Computing DOS...")
omega_kpm, dos_kpm = kpm_dos(H; N_randvec=3, N_moments=N_moments,
                              N_points=2*N_moments, epsilon=0.05)

# ===== Hall conductance =====
println("Computing KPM σ_xy, σ_xx ...")
E_fermi = range(-4.0, 4.0; length=400) |> collect

# Honeycomb prefactor correction (2 atoms/cell, A_cell = √3)
# kpm_hall_conductance uses 4/a² for 1-atom square; for honeycomb multiply by (2/√3)
# via the a_scale parameter such that a_scale² = √3/2 ⟹ use a_scale = (√3/2)^(1/2)

t_kpm = @elapsed begin
    sigma_xy, sigma_xx = kpm_hall_conductance(H, Jx, Jy, E_fermi;
        N_moments = N_moments,
        N_randvec = N_randvec,
        kernel    = "Jackson",
        epsilon   = 0.05)
end
# kpm_hall_conductance uses prefactor 4/a² for 1-atom/cell (A_cell=1).
# Haldane honeycomb: 2 atoms/cell, A_cell=√3  →  multiply by (2/√3)
honeycomb_correction = 2 / sqrt(3)
sigma_xy .*= honeycomb_correction
sigma_xx .*= honeycomb_correction
println("Done in $(round(t_kpm; digits=1)) s")

# ===== Plot =====
p_dos  = plot_dos(omega_kpm, dos_kpm; title = "Haldane model DOS")
p_cond = plot_conductance(E_fermi, sigma_xy, sigma_xx;
                          plateaux = [-1, 0, 1],
                          title = "Haldane: $(Lx)×$(Ly), t₂=$(t2), M=$(M_sub), M=$(N_moments)")

display(p_dos)
display(p_cond)
savefig(p_dos,  joinpath(@__DIR__, "haldane_dos.png"))
savefig(p_cond, joinpath(@__DIR__, "haldane_conductance.png"))
println("Saved haldane_dos.png and haldane_conductance.png")
