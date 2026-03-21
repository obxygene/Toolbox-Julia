# validate_conductance_square.jl
# Small-system validation: KPM σ_xy vs exact diagonalization for the Hofstadter
# square lattice model.  Mean KPM/Exact ratio should be ≈ 1.0.
#
# Reference: Validate_KPM_Conductance_Square.m

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using KernelPolynomialMethod
using SparseArrays
using LinearAlgebra
using Plots

# ===== Parameters =====
t   = 1.0
q   = 5            # flux denominator
Lx  = q * 4        # 20 × 4 = 20 (multiple of q)
Ly  = 20
phi = 1/q
N_sites = Lx * Ly

println("=== Validate KPM vs Exact  (Square, Hofstadter) ===")
println("$(Lx)×$(Ly) sites, ϕ = 1/$q")

# ===== Build Hofstadter Hamiltonian (PBC) =====
site = (ix, iy) -> mod(iy-1, Ly)*Lx + mod(ix-1, Lx) + 1

h_rows  = Int[]; h_cols  = Int[]; h_vals  = ComplexF64[]
jx_rows = Int[]; jx_cols = Int[]; jx_vals = ComplexF64[]
jy_rows = Int[]; jy_cols = Int[]; jy_vals = ComplexF64[]

for iy in 1:Ly, ix in 1:Lx
    s1 = site(ix, iy)

    # x-hop
    s2  = site(ix+1, iy); hop = -t
    push!(h_rows, s1); push!(h_cols, s2); push!(h_vals, hop)
    push!(h_rows, s2); push!(h_cols, s1); push!(h_vals, conj(hop))
    push!(jx_rows, s1); push!(jx_cols, s2); push!(jx_vals,  1im*hop)
    push!(jx_rows, s2); push!(jx_cols, s1); push!(jx_vals,  1im*conj(hop)*(-1))
    push!(jy_rows, s1); push!(jy_cols, s2); push!(jy_vals, 0im)
    push!(jy_rows, s2); push!(jy_cols, s1); push!(jy_vals, 0im)

    # y-hop with Peierls phase
    s3    = site(ix, iy+1)
    phase = exp(1im * 2π * phi * ix)
    hop   = -t * phase
    push!(h_rows, s1); push!(h_cols, s3); push!(h_vals, hop)
    push!(h_rows, s3); push!(h_cols, s1); push!(h_vals, conj(hop))
    push!(jx_rows, s1); push!(jx_cols, s3); push!(jx_vals, 0im)
    push!(jx_rows, s3); push!(jx_cols, s1); push!(jx_vals, 0im)
    push!(jy_rows, s1); push!(jy_cols, s3); push!(jy_vals,  1im*hop)
    push!(jy_rows, s3); push!(jy_cols, s1); push!(jy_vals,  1im*conj(hop)*(-1))
end

H_sp  = sparse(h_rows,  h_cols,  h_vals,  N_sites, N_sites)
Jx_sp = sparse(jx_rows, jx_cols, jx_vals, N_sites, N_sites)
Jy_sp = sparse(jy_rows, jy_cols, jy_vals, N_sites, N_sites)

println("H Hermitian check: $(round(norm(H_sp-H_sp',Inf)/max(1,norm(H_sp,Inf)); sigdigits=2))")

# ===== Exact diagonalisation =====
println("\nExact diagonalisation ($N_sites sites)...")
t_ed = @elapsed begin
    F = eigen(Hermitian(Matrix(H_sp)))
    eigvals = F.values
    V = F.vectors
end
println("Done in $(round(t_ed; digits=1)) s.  Bandwidth: [$(round(eigvals[1];digits=4)), $(round(eigvals[end];digits=4))]")

# ===== Exact σ_xy via Kubo-Blount formula =====
# σ_xy/(e²/h) = (2π/Ω) Σ_{n occ, m unocc} 2 Im[⟨n|Jx|m⟩⟨m|Jy|n⟩] / (Em−En)²
Ω = Float64(N_sites)   # A_cell = 1 for square lattice
Jx_eig = V' * Jx_sp * V
Jy_eig = V' * Jy_sp * V

E_fermi = range(-4.5, 4.5; length=200) |> collect
sigma_xy_exact = zeros(length(E_fermi))

println("Computing exact σ_xy at $(length(E_fermi)) Fermi energies...")
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
        sigma_xy_exact[ii] = (2π / Ω) * S
    end
end
println("Done in $(round(t_exact; digits=1)) s")

# ===== KPM σ_xy =====
println("\nComputing KPM σ_xy...")
t_kpm = @elapsed begin
    sigma_xy_kpm, sigma_xx_kpm = kpm_hall_conductance(H_sp, Jx_sp, Jy_sp, E_fermi;
        N_moments = 256,
        N_randvec = 20,
        kernel    = "Jackson",
        epsilon   = 0.05)
end
println("Done in $(round(t_kpm; digits=1)) s")

# ===== Ratio calibration =====
signif = findall(abs.(sigma_xy_exact) .> 0.5)
if !isempty(signif)
    ratio = sigma_xy_kpm[signif] ./ sigma_xy_exact[signif]
    valid = isfinite.(ratio) .& (abs.(ratio) .< 100)
    r = ratio[valid]
    println("\n=== Square Lattice: Prefactor Calibration ===")
    println("Mean  KPM/Exact ratio : $(round(mean(r); digits=4))")
    println("Median KPM/Exact ratio: $(round(median(r); digits=4))")
    println("Std of ratio          : $(round(std(r); digits=4))")
    println("(Target ≈ 1.0)")
end

# ===== Plot =====
p = plot_conductance_compare(E_fermi, sigma_xy_kpm, sigma_xy_exact,
                              sigma_xx_kpm, zeros(length(E_fermi));
                              plateaux = -4:4,
                              title = "Square Hofstadter $(Lx)×$(Ly), ϕ=1/$q: KPM vs Exact")
display(p)
savefig(p, joinpath(@__DIR__, "validate_square.png"))
println("Saved validate_square.png")
