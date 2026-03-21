"""
Monatomic_Transmission_Analytical_Comparison.jl

Numerically computes the transmission of a monatomic tight-binding wire
using the Recursive Green Function method and compares it against the
analytical single-channel result, summed over transverse modes.

Mirrors the MATLAB script Monatomic_Lattice_Transmission_Analytical.m.

Physics
-------
- 2D strip: Width W transverse sites, `layer` longitudinal unit cells.
- Tight-binding: on-site ε₀=0, nearest-neighbour hopping t (both x and y).
- Interface barrier: lead-to-central hopping tr (may differ from t).
- Transverse modes: Eₙ = -2t·cos(nπ/(W+1)),  n = 1 … W
- Analytical single-channel transmission (per mode, shifted to Eₙ):
    T_1ch(ω) = (|ω| < 2t) *
               t⁴ tr⁴ (4t²-ω²) /
               [(t²ω²+tr⁴-tr²ω²)(4t⁶+ω²(-4t⁴+2t²tr²+tr⁴)+ω⁴(t²-tr²))]
- Total: T(ω) = Σₙ T_1ch(ω - Eₙ)

References
----------
M.P. Lopez Sancho et al., J. Phys. F: Met. Phys. 15, 851 (1985).
D.S. Fisher & P.A. Lee, Phys. Rev. B 23, 6851 (1981).
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()
using RecursiveGreenFunction
using LinearAlgebra
using Printf
using Plots

# ── Parameters (match MATLAB reference) ─────────────────────────────────────
const Width   = 10       # transverse sites W
const t       = 6.5      # intra-layer hopping
const tr      = 6.5      # lead–central interface hopping
const layer   = 3        # number of central layers
const eta     = 1e-5     # broadening η
const N_point = 101      # energy grid points
const accuracy = 1e-10   # Lopez-Sancho convergence threshold

# ── Hamiltonians ─────────────────────────────────────────────────────────────
# Intra-layer Hamiltonian H₀₀ (W×W tridiagonal, on-site ε₀=0)
H00 = zeros(Float64, Width, Width)
for i in 1:Width-1
    H00[i, i+1] = -t
    H00[i+1, i] = -t
end
const H00_m = H00

# Inter-layer hopping V = -t·I (pure longitudinal hopping)
const V = -t * Matrix{Float64}(I, Width, Width)

# Lead–central coupling (scaled by tr/t)
const V_lr = (tr/t) * V

# ── Energy grid (slightly outside [-2t, 2t] to show band edges) ─────────────
const omega_vec = range(-2*t*1.01, 2*t*1.01, N_point)

# ── Transverse mode eigenvalues ───────────────────────────────────────────────
const EigenEnergy = [-2t * cos(n*π/(Width+1)) for n in 1:Width]

# ── Analytical single-channel transmission ────────────────────────────────────
function T_1ch(ω::Real, t::Real, tr::Real)
    abs(ω) >= 2t && return 0.0
    num   = t^4 * tr^4 * (4t^2 - ω^2)
    den1  = t^2 * ω^2 + tr^4 - tr^2 * ω^2
    den2  = 4t^6 + ω^2*(-4t^4 + 2t^2*tr^2 + tr^4) + ω^4*(t^2 - tr^2)
    d = den1 * den2
    d ≈ 0 && return 0.0
    return num / d
end

function T_analytical(ω::Real)
    return sum(T_1ch(ω - En, t, tr) for En in EigenEnergy)
end

# ── Numerical RGF computation ─────────────────────────────────────────────────
T_numerical  = zeros(Float64, N_point)
T_analytic   = zeros(Float64, N_point)

println("Computing RGF transmission for $N_point energy points...")
t_start = time()

for (ii, ω) in enumerate(omega_vec)
    # Surface GFs (right lead: forward V; left lead: backward V')
    G_00_R, _ = surface_green_function_v2(H00_m, V,  ω, accuracy; eta=eta)
    G_00_L, _ = surface_green_function_v2(H00_m, V', ω, accuracy; eta=eta)

    # Self-energies from leads (interface coupling tr)
    Sigma_L = surface_green_function_self_energy(G_00_L, V_lr')
    Sigma_R = surface_green_function_self_energy(G_00_R, V_lr)

    # Level-width (broadening) matrices
    Gamma_L = surface_green_function_broadening(Sigma_L)
    Gamma_R = surface_green_function_broadening(Sigma_R)

    # Recursive GF G_{1L}
    G_1L = recursive_green_function_1l(H00_m, V, layer, ω, Sigma_L, Sigma_R; eta=eta)

    # Fisher-Lee transmission
    T_numerical[ii] = real(green_function_transmission(Gamma_L, Gamma_R, G_1L))

    # Analytical
    T_analytic[ii]  = T_analytical(ω)
end

t_end = time()
@printf("Done in %.2f s\n", t_end - t_start)

# ── Error metric ──────────────────────────────────────────────────────────────
max_err  = maximum(abs.(T_numerical - T_analytic))
mean_err = sum(abs.(T_numerical - T_analytic)) / N_point
@printf("Max  |T_num - T_ana| = %.2e\n", max_err)
@printf("Mean |T_num - T_ana| = %.2e\n", mean_err)

# ── Plots ─────────────────────────────────────────────────────────────────────
ω_vals = collect(omega_vec)

p1 = plot(ω_vals, T_numerical;
    label="Numerical (RGF)",
    xlabel="Energy ω", ylabel="Transmission T(ω)",
    title="RGF: W=$Width, L=$layer, t=$t, tᵣ=$tr",
    lw=1.5, color=:blue, marker=:circle, markersize=2)

p2 = plot(ω_vals, T_analytic;
    label="Analytical",
    xlabel="Energy ω", ylabel="Transmission T(ω)",
    title="Analytical: Σₙ T₁ch(ω−Eₙ)",
    lw=1.5, color=:red)

p3 = plot(ω_vals, T_numerical;
    label="RGF", lw=1.5, color=:blue, marker=:circle, markersize=2)
plot!(p3, ω_vals, T_analytic;
    label="Analytical", lw=1.5, color=:red, linestyle=:dash,
    xlabel="Energy ω", ylabel="Transmission T(ω)",
    title="Comparison (max err = $(@sprintf("%.1e", max_err)))")

plt = plot(p1, p2, p3; layout=(1,3), size=(1200, 400))
display(plt)

savefig(plt, joinpath(@__DIR__, "Monatomic_Transmission_Comparison.png"))
println("Plot saved to examples/Monatomic_Transmission_Comparison.png")
