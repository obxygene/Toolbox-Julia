# benchmark_dos_square.jl
# KPM DOS for a 2D square tight-binding lattice, compared with the analytic result.
# Analytic DOS: ρ(E) ∝ K(√(1 − E²/16)) / π  where K is the complete elliptic integral.
#
# Reference: Weisse et al., Rev. Mod. Phys. 78, 275 (2006)

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using KernelPolynomialMethod
using SparseArrays
using LinearAlgebra
using SpecialFunctions   # for ellipk
using Plots

# ===== Parameters =====
Lx = 1000
Ly = 1000
t  = 1.0
N_sites = Lx * Ly

N_randvec = 3
N_moments = 1000
N_points  = 2 * N_moments

println("System: $(Lx)×$(Ly) square lattice, N = $N_sites sites")
println("KPM: $N_moments moments, $N_randvec random vectors")

# ===== Build nearest-neighbour Hamiltonian (PBC) =====
println("Building Hamiltonian...")

site = (ix, iy) -> (mod(iy-1, Ly))*Lx + mod(ix-1, Lx) + 1

rows = Int[]; cols = Int[]; vals = ComplexF64[]

for iy in 1:Ly, ix in 1:Lx
    s1 = site(ix, iy)
    # x-hop
    s2 = site(ix+1, iy)
    push!(rows, s1); push!(cols, s2); push!(vals, -t)
    push!(rows, s2); push!(cols, s1); push!(vals, -t)
    # y-hop
    s3 = site(ix, iy+1)
    push!(rows, s1); push!(cols, s3); push!(vals, -t)
    push!(rows, s3); push!(cols, s1); push!(vals, -t)
end

H = sparse(rows, cols, vals, N_sites, N_sites)
println("Hermitian check: $(norm(H - H', Inf))")

# ===== KPM DOS =====
println("Computing KPM DOS...")
t_kpm = @elapsed begin
    omega_kpm, dos_kpm = kpm_dos(H;
        N_randvec = N_randvec,
        N_moments = N_moments,
        N_points  = N_points,
        epsilon   = 0.05)
end
println("Done in $(round(t_kpm; digits=1)) s")

# ===== Analytic DOS (2D square lattice) =====
# ρ(E) = (1/π²) K(√(1 − E²/16))  for |E| < 4, per site, normalised to 1
println("Computing analytic DOS...")
e_analytic = range(-3.98, 3.98; length = 800)
dos_analytic = map(e_analytic) do e
    k2 = 1.0 - (e/4)^2          # k² for ellipk
    k2 < 0 && return 0.0
    ellipk(sqrt(k2)) / π^2
end
# Normalise
Δe = step(e_analytic)
dos_analytic ./= sum(dos_analytic) * Δe

# ===== Plot =====
p = plot_dos_compare(omega_kpm, dos_kpm, collect(e_analytic), dos_analytic;
                     title = "2D square lattice DOS: KPM vs Analytic")
display(p)
savefig(p, joinpath(@__DIR__, "benchmark_dos_square.png"))
println("Saved benchmark_dos_square.png")
