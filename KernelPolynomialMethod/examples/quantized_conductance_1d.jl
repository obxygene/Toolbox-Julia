# quantized_conductance_1d.jl
#
# Quantized Landauer conductance of a quasi-1D tight-binding strip via KPM.
#
# Physics
# ───────
# W-wide strip, hopping t, PBC transverse and longitudinal:
#   ε(kx, ky) = −2t cos kx − 2t cos ky
# Transverse modes: ky_n = 2πn/W,  band centre ε_n = −2t cos(ky_n).
# Mode n is open (propagating) when  ε_n − 2t < EF < ε_n + 2t.
#
# Landauer conductance (Fisher-Lee / Büttiker):
#   G(EF) = N_ch(EF) × e²/h
#
# KPM role
# ────────
# kpm_dos → ρ(E) without diagonalisation.
# Each conductance step coincides with a Van Hove singularity:
#   ρ ∝ 1/√(4t²−(E−ε_n)²)  diverges at sub-band edges E = ε_n ± 2t.
# KPM thus directly *encodes* the step locations in its spectral output.
#
# Connection to RecursiveGreenFunction in this repo:
#   Fisher-Lee: T(E) = Tr[ΓL G^R ΓR G^A] = N_ch(E) for a clean strip.
#   Both approaches see the same quantized steps at the same energies.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using KernelPolynomialMethod
using SparseArrays
using LinearAlgebra
using Printf
using Plots

# ────────────────────────────────────────────────────────────────────────────
# Parameters
# ────────────────────────────────────────────────────────────────────────────
t  = 1.0    # hopping
W  = 8      # transverse sites  (up to W open channels)
L  = 60     # longitudinal sites (PBC)
N  = W * L

println("=== Quantized Conductance  (W=$W × L=$L, N=$N sites, t=$t) ===")

# ────────────────────────────────────────────────────────────────────────────
# Hamiltonian (PBC both directions)
#   site: s = (iy−1)×L + ix
# ────────────────────────────────────────────────────────────────────────────
site = (ix, iy) -> (mod(iy - 1, W)) * L + mod(ix - 1, L) + 1

rH = Int[]; cH = Int[]; vH = Float64[]

for iy in 1:W, ix in 1:L
    s1 = site(ix, iy)
    s2 = site(ix + 1, iy);  push!(rH,s1);push!(cH,s2);push!(vH,-t);  push!(rH,s2);push!(cH,s1);push!(vH,-t)
    s3 = site(ix, iy + 1);  push!(rH,s1);push!(cH,s3);push!(vH,-t);  push!(rH,s3);push!(cH,s1);push!(vH,-t)
end

H = sparse(rH, cH, vH, N, N)
@assert norm(H - H', Inf) < 1e-12 "H not Hermitian"
println("H built: $(nnz(H)) nonzeros,  Hermitian ✓")

# ────────────────────────────────────────────────────────────────────────────
# Sub-band structure and channel counting (analytical, O(W))
# ────────────────────────────────────────────────────────────────────────────
ε_modes = [-2t * cos(2π * n / W) for n in 0:(W-1)]

println("\nSub-band centres ε_n and channel windows [ε_n−2t, ε_n+2t]:")
for εn in sort(ε_modes)
    @printf("  ε=%+.4f   window (%+.4f, %+.4f)\n", εn, εn-2t, εn+2t)
end

n_channels(E) = sum(εn - 2t < E < εn + 2t  for εn in ε_modes)
channel_edges = sort(unique(vcat([εn - 2t for εn in ε_modes],
                                 [εn + 2t for εn in ε_modes])))
channel_edges = filter(e -> -4.15 < e < 4.15, channel_edges)

E_axis  = collect(range(-4.15, 4.15; length=600))
G_exact = Float64[n_channels(E) for E in E_axis]

println("\nConductance steps at E = ", round.(channel_edges; digits=4))

# ────────────────────────────────────────────────────────────────────────────
# KPM density of states  ρ(E) = (1/N) Tr[δ(E−H)]
# Van Hove singularities at sub-band edges = conductance step positions
# ────────────────────────────────────────────────────────────────────────────
println("\nKPM DOS …")
t_dos = @elapsed ω_dos, ρ_dos = kpm_dos(H;
    N_moments = 512,
    N_randvec = 30,
    N_points  = 2000,
    kernel    = "Jackson",
    bounds    = (-4.2, 4.2))
println("  done in $(round(t_dos; digits=2)) s")

# Analytic 1D DOS (Van Hove, per strip):  ρ_n(E) = 1/(π√(4t²−(E−εn)²))
# For the full W-wide strip (normalised per site): ρ_strip = (1/W) Σ_n ρ_n
η_vH  = 0.08 * t   # broadening for analytic curve
ρ_analytic = [sum(1 / (π * sqrt(max((2t)^2 - (E - εn)^2, η_vH^2)))
                  for εn in ε_modes) / W
              for E in E_axis]

# ────────────────────────────────────────────────────────────────────────────
# Plot  – three stacked panels
# ────────────────────────────────────────────────────────────────────────────
XLIMS  = (-4.2, 4.2)
CLR_S  = :steelblue
CLR_E  = :black
CLREDG = :crimson

# Panel 1: KPM DOS
p1 = plot(ω_dos, ρ_dos;
          ylabel = "ρ (per site, t⁻¹)",
          title  = "KPM density of states  —  W=$W strip",
          label  = "KPM DOS",
          color  = CLR_S, lw = 1.3,
          xlims  = XLIMS, ylims = (0, maximum(ρ_dos) * 1.15),
          legend = :topleft, grid = true, xlabel = "")
plot!(p1, E_axis, ρ_analytic;
      label = "Analytic (broadened)",
      color = :orange, lw = 1.1, ls = :dash, alpha = 0.85)
vline!(p1, channel_edges; lw = 0.8, ls = :dot, color = CLREDG, label = "sub-band edges")

# Panel 2: N_ch staircase (= quantized conductance in e²/h)
p2 = plot(E_axis, G_exact;
          xlabel = "Fermi energy  EF / t",
          ylabel = "G  (e²/h)",
          title  = "Landauer conductance  G = N_ch × e²/h",
          label  = "G = N_ch(EF)   [exact band count]",
          color  = CLR_E, lw = 2.2, ls = :dash,
          xlims  = XLIMS, ylims = (-0.4, W + 0.5),
          legend = :topleft, grid = true)
hline!(p2, 0:W; lw = 0.4, ls = :dot, color = :gray, label = nothing)
vline!(p2, channel_edges; lw = 0.8, ls = :dot, color = CLREDG, label = nothing)

# Annotate each plateau
for nc in 1:W
    idx = findall(G_exact .== nc)
    isempty(idx) && continue
    Emid = E_axis[idx[div(length(idx), 2)]]
    annotate!(p2, Emid, nc + 0.38, text("$nc", 7, :center, :navy))
end

p = plot(p1, p2; layout = (2, 1), size = (860, 640),
         left_margin = 5Plots.mm, bottom_margin = 3Plots.mm)
display(p)
outfile = joinpath(@__DIR__, "quantized_conductance_1d.png")
savefig(p, outfile)
println("Saved: $outfile")

# ────────────────────────────────────────────────────────────────────────────
# Summary table
# ────────────────────────────────────────────────────────────────────────────
println("\n══ Conductance quantisation summary ══════════════════════════════")
println("  G (e²/h) │  EF range              │  Van Hove peak in ρ at")
for nc in 1:W
    idx = findall(G_exact .== nc)
    isempty(idx) && continue
    Elo = E_axis[first(idx)];  Ehi = E_axis[last(idx)]
    # find the corresponding step edge (channel that just opened)
    step_edge = filter(e -> abs(e - Elo) < 0.15, channel_edges)
    edge_str  = isempty(step_edge) ? "—" : join(round.(step_edge; digits=3), ", ")
    @printf("    %2d     │  (%+.3f, %+.3f)   │  %s\n", nc, Elo, Ehi, edge_str)
end
println("══════════════════════════════════════════════════════════════════")
println("\nConclusion: G steps by e²/h each time EF crosses a sub-band edge.")
println("KPM DOS diverges (Van Hove) at the same energies → the spectral")
println("structure directly reveals the quantized conductance steps.")
println("Same steps seen by RGF Fisher-Lee: T = Tr[ΓL G^R ΓR G^A].")
