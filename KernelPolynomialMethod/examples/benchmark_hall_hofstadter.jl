# benchmark_hall_hofstadter.jl
# KPM Hall conductance for the Hofstadter model (2D square lattice + B field).
# Landau gauge A=(0,Bx,0), Peierls phase on y-hoppings.
# Expected: σ_xy = integer × e²/h in Hofstadter gaps.
#
# Requirements:
#   - Lx must be a multiple of q (flux denominator) for PBC gauge consistency.
#
# Reference: Garcia et al., Phys. Rev. B 91, 245140 (2015)

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using KernelPolynomialMethod
using SparseArrays
using LinearAlgebra
using Plots

# ===== Parameters =====
phi = 1//9           # flux per plaquette = p/q
q   = denominator(phi)
Lx  = q * 50         # must be multiple of q
Ly  = 360
t   = 1.0
N_sites = Lx * Ly

N_moments = 256
N_randvec = 5

println("Hofstadter: $(Lx)×$(Ly) square lattice, ϕ = $phi, N = $N_sites")
println("KPM: $N_moments moments, $N_randvec random vectors")

# ===== Build Hamiltonian and current operators =====
println("Building H, Jx, Jy ...")

site = (ix, iy) -> mod(iy-1, Ly)*Lx + mod(ix-1, Lx) + 1

h_rows  = Int[]; h_cols  = Int[]; h_vals  = ComplexF64[]
jx_rows = Int[]; jx_cols = Int[]; jx_vals = ComplexF64[]
jy_rows = Int[]; jy_cols = Int[]; jy_vals = ComplexF64[]

ϕ = Float64(phi)

for iy in 1:Ly, ix in 1:Lx
    s1 = site(ix, iy)

    # x-hop (no Peierls phase in Landau gauge)
    s2 = site(ix+1, iy)
    hop = -t
    push!(h_rows, s1); push!(h_cols, s2); push!(h_vals, hop)
    push!(h_rows, s2); push!(h_cols, s1); push!(h_vals, conj(hop))
    push!(jx_rows, s1); push!(jx_cols, s2); push!(jx_vals,  1im*hop*1)
    push!(jx_rows, s2); push!(jx_cols, s1); push!(jx_vals,  1im*conj(hop)*(-1))
    push!(jy_rows, s1); push!(jy_cols, s2); push!(jy_vals, 0im)
    push!(jy_rows, s2); push!(jy_cols, s1); push!(jy_vals, 0im)

    # y-hop with Peierls phase exp(i 2π ϕ ix)
    s3   = site(ix, iy+1)
    phase = exp(1im * 2π * ϕ * ix)
    hop   = -t * phase
    push!(h_rows, s1); push!(h_cols, s3); push!(h_vals, hop)
    push!(h_rows, s3); push!(h_cols, s1); push!(h_vals, conj(hop))
    push!(jx_rows, s1); push!(jx_cols, s3); push!(jx_vals, 0im)
    push!(jx_rows, s3); push!(jx_cols, s1); push!(jx_vals, 0im)
    push!(jy_rows, s1); push!(jy_cols, s3); push!(jy_vals,  1im*hop*1)
    push!(jy_rows, s3); push!(jy_cols, s1); push!(jy_vals,  1im*conj(hop)*(-1))
end

H  = sparse(h_rows,  h_cols,  h_vals,  N_sites, N_sites)
Jx = sparse(jx_rows, jx_cols, jx_vals, N_sites, N_sites)
Jy = sparse(jy_rows, jy_cols, jy_vals, N_sites, N_sites)

println("H  Hermitian check: $(round(norm(H  - H',  Inf); sigdigits=2))")
println("Jx Hermitian check: $(round(norm(Jx - Jx', Inf); sigdigits=2))")
println("Jy Hermitian check: $(round(norm(Jy - Jy', Inf); sigdigits=2))")

# ===== KPM DOS =====
println("Computing DOS...")
omega_kpm, dos_kpm = kpm_dos(H; N_randvec=3, N_moments=N_moments,
                              N_points=2*N_moments, epsilon=0.05)

# ===== KPM Hall conductance =====
println("Computing KPM σ_xy, σ_xx ...")
E_fermi = range(-4.5, 4.5; length=300) |> collect

t_kpm = @elapsed begin
    sigma_xy, sigma_xx = kpm_hall_conductance(H, Jx, Jy, E_fermi;
        N_moments = N_moments,
        N_randvec = N_randvec,
        kernel    = "Jackson",
        epsilon   = 0.05)
end
println("Done in $(round(t_kpm; digits=1)) s")

# ===== Plot =====
p_dos = plot_dos(omega_kpm, dos_kpm;
                 title = "Hofstadter DOS  ϕ = $phi")
p_cond = plot_conductance(E_fermi, sigma_xy, sigma_xx;
                          plateaux = -4:4,
                          title = "Hofstadter IQHE: $(Lx)×$(Ly), ϕ=$(ϕ), M=$N_moments, R=$N_randvec")

display(p_dos)
display(p_cond)
savefig(p_dos,  joinpath(@__DIR__, "hofstadter_dos.png"))
savefig(p_cond, joinpath(@__DIR__, "hofstadter_conductance.png"))
println("Saved hofstadter_dos.png and hofstadter_conductance.png")
