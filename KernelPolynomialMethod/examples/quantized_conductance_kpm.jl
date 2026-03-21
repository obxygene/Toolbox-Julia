# quantized_conductance_kpm.jl
#
# Quantized Landauer conductance via KPM — large-scale demonstration.
#
# Physics
# ───────
# Landauer conductance:  G(E_F) = N_ch(E_F) × e²/h
#
# Each conductance step occurs when E_F crosses a 1D sub-band edge
# E = ε_n ± 2t.  At these energies the 1D DOS diverges (Van Hove
# singularity): ρ ∝ 1/√(4t²-(E-ε_n)²).
#
# KPM role
# ────────
# KPM computes ρ(E) without diagonalisation via stochastic Chebyshev
# expansion.  The Van Hove peaks in ρ(E) mark precisely the energies
# where G steps by e²/h.  Integrating ρ(E) mode-by-mode gives the
# open-channel count directly from the spectral function.
#
# Demonstration:
#   1. Build a large quasi-1D strip  (W × L sites)
#   2. KPM DOS  →  shows W-fold Van Hove divergences
#   3. N_ch(E) staircase (from KPM mode-projected DOS threshold)
#   4. Compare with exact analytical channel count

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using KernelPolynomialMethod
using SparseArrays
using LinearAlgebra
using Printf
using Plots

# ─────────────────────────────────────────────────────────────────────────────
# Parameters  (larger scale than the 1D intro example)
# ─────────────────────────────────────────────────────────────────────────────
t      = 1.0
W      = 8          # transverse sites  (up to W channels)
L      = 800        # longitudinal sites (PBC in x)
N      = W * L
N_mom  = 2048
N_rnd  = 30

println("=== KPM Quantized Conductance  W=$W × L=$L, N=$N sites ===")

# ─────────────────────────────────────────────────────────────────────────────
# Hamiltonian  (PBC both x and y)
#   site: s = (iy−1)×L + ix
# ─────────────────────────────────────────────────────────────────────────────
site = (ix, iy) -> (mod(iy - 1, W)) * L + mod(ix - 1, L) + 1

rH = Int[]; cH = Int[]; vH = Float64[]
for iy in 1:W, ix in 1:L
    s1 = site(ix, iy)
    s2 = site(ix + 1, iy); push!(rH,s1);push!(cH,s2);push!(vH,-t);  push!(rH,s2);push!(cH,s1);push!(vH,-t)
    s3 = site(ix, iy + 1); push!(rH,s1);push!(cH,s3);push!(vH,-t);  push!(rH,s3);push!(cH,s1);push!(vH,-t)
end
H = sparse(rH, cH, vH, N, N)
@assert norm(H - H', Inf) < 1e-12 "H not Hermitian"
println("H: $(nnz(H)) nonzeros, Hermitian ✓")

# ─────────────────────────────────────────────────────────────────────────────
# Sub-band structure (analytical, O(W))
# ─────────────────────────────────────────────────────────────────────────────
ε_modes = [-2t * cos(2π * n / W) for n in 0:(W-1)]   # transverse mode energies

# Number of open channels at each energy (exact, O(W))
n_channels(E) = sum(εn - 2t < E < εn + 2t  for εn in ε_modes)

# Sub-band edges = positions of Van Hove singularities = conductance step energies
band_edges = sort(unique(vcat(ε_modes .- 2t, ε_modes .+ 2t)))
band_edges = filter(e -> -4.05 < e < 4.05, band_edges)

println("\nSub-band centres (W=$W modes):")
for ε in sort(unique(round.(ε_modes; digits=4)))
    cnt = count(≈(ε; atol=1e-8), ε_modes)
    @printf("  ε = %+.4f  (×%d)  window (%+.4f, %+.4f)\n",
            ε, cnt, ε-2t, ε+2t)
end

# ─────────────────────────────────────────────────────────────────────────────
# KPM density of states  ρ(E) = (1/N) Tr[δ(E−H)]
# Van Hove singularities diverge at sub-band edges
# ─────────────────────────────────────────────────────────────────────────────
println("\nKPM DOS (N_mom=$N_mom, N_rnd=$N_rnd, N=$N sites)…")
t_dos = @elapsed ω_dos, ρ_dos = kpm_dos(H;
    N_moments = N_mom,
    N_randvec = N_rnd,
    N_points  = 4000,
    kernel    = "Jackson",
    bounds    = (-4.2, 4.2))
println("  done in $(round(t_dos; digits=1)) s")

# ─────────────────────────────────────────────────────────────────────────────
# KPM-based channel count via mode-projected DOS
#
# For each transverse mode n with centre εₙ, the 1D sub-band contributes
#   ρₙ(E) = 1/(πW√(4t²−(E−εₙ)²))  for |E−εₙ| < 2t
#
# KPM resolves this via the operator moments of Pₙ = projector onto mode n:
#   μₙ,k = Tr[Pₙ Tₖ(H̃)] / N   →   ρₙ(E) via DCT reconstruction
#
# An open channel at energy E is identified when the mode-projected DOS
# exceeds the van Hove threshold (> 1/π × 1/(2t × W) = min bandwidth DOS).
# ─────────────────────────────────────────────────────────────────────────────
println("Mode-projected KPM DOS …")
t_mode = @elapsed begin
    E_ax    = collect(range(-4.15, 4.15; length=1200))
    G_exact = Float64[n_channels(E) for E in E_ax]

    # KPM mode-projected DOS via kpm_moments_operator
    H_sc, a_sc, b_sc = kpm_scale_hamiltonian(H; bounds=(-4.2t, 4.2t))
    N_mom_mode = 1024   # sufficient for mode projection

    # Build transverse projectors Pₙ (each selects 1 out of W modes)
    # φₙ(iy) = (1/√W) exp(2πi n (iy-1)/W)
    # Pₙ_sites: W×L projector onto mode n (outer product over x)
    ρ_modes = zeros(length(E_ax), W)

    m_vec = 0:(N_mom_mode - 1)
    g_J_mode = kpm_kernel(N_mom_mode; kernel="Jackson")

    for n in 0:(W-1)
        # Mode-n projector: diagonal N×N matrix with entries |φₙ(iy)|² = 1/W
        # (uniform in both x and y — projection gives bulk mode n LDOS / W)
        # Use the seed vector v = (1/√N) exp(2πin(iy-1)/W) (all x uniform)
        v_mode = zeros(ComplexF64, N)
        for iy in 1:W, ix in 1:L
            s = site(ix, iy)
            v_mode[s] = exp(2π*im*n*(iy-1)/W) / sqrt(N)
        end

        # 1D moments:  μₖ = ⟨v_mode | Tₖ(H̃) | v_mode⟩  (not stochastic — exact)
        mu = zeros(ComplexF64, N_mom_mode)
        αm, α = v_mode, H_sc * v_mode
        mu[1] = dot(v_mode, αm)
        mu[2] = dot(v_mode, α)
        for k in 3:N_mom_mode
            αp = 2 .* (H_sc * α) .- αm
            mu[k] = dot(v_mode, αp)
            αm, α = α, αp
        end
        mu_r = real.(mu)   # should be real for Hermitian H and real-symmetric v

        # KPM reconstruction at each energy
        gc = copy(g_J_mode);  gc[1] /= 2
        mu_K = gc .* mu_r
        for (ie, E) in enumerate(E_ax)
            Esc = clamp((E - b_sc) / a_sc, -1 + 0.01, 1 - 0.01)
            T_EF = cos.(m_vec .* acos(Esc))
            ρ_modes[ie, n+1] = real(dot(T_EF, mu_K)) / (π * a_sc * sqrt(1 - Esc^2))
        end
    end

    # Total KPM DOS (sum over modes)
    ρ_total_kpm = sum(ρ_modes; dims=2)[:]

    # Channel count from mode-projected DOS threshold
    # A mode n is "open" at energy E if its DOS exceeds the minimum bulk value
    # ρ_n^min = 1/(π × 2t × W) (value at sub-band centre)
    rho_threshold = 1.0 / (π * 2t * W) * 0.5   # 50% of centre-band DOS
    G_kpm = sum(ρ_modes .> rho_threshold, dims=2)[:]
end
println("  done in $(round(t_mode; digits=1)) s")

# Analytic 1D DOS for comparison
ρ_analytic = [sum(1/(π * max(sqrt(max((2t)^2 - (E-εn)^2, 0.0)), 0.05t))
                  for εn in ε_modes) / W
              for E in E_ax]

# ─────────────────────────────────────────────────────────────────────────────
# Plateau verification
# ─────────────────────────────────────────────────────────────────────────────
println("\n── Plateau check (E_F at plateau centres) ─────────────────────────")
println("  N_ch │ E_F(mid) │ G_KPM  │ N_ch_exact")
for nc in 0:W
    idx = findall(G_exact .== nc);  isempty(idx) && continue
    # first contiguous run
    run_end = idx[1]
    for k in 2:length(idx); idx[k] == idx[k-1]+1 ? (run_end = idx[k]) : break; end
    mid = div(idx[1] + run_end, 2)
    @printf("   %2d  │  %+6.3f  │   %2d   │  %2d\n",
            nc, E_ax[mid], G_kpm[mid], nc)
end
println("──────────────────────────────────────────────────────────────────────")

# ─────────────────────────────────────────────────────────────────────────────
# Plot — two panels
# ─────────────────────────────────────────────────────────────────────────────
XLIMS  = (-4.2, 4.2)
CLR_S  = :steelblue
CLR_E  = :black
CLR_ED = :crimson
CLR_KPM = :darkorange

# Panel 1: KPM DOS with Van Hove singularities
ρ_max = min(maximum(ρ_dos), 3.0)   # cap for display
p1 = plot(ω_dos, min.(ρ_dos, ρ_max);
          ylabel = "ρ (per site · t⁻¹)",
          title  = "KPM DOS  —  W=$W strip, N=$N, M=$N_mom moments",
          label  = "KPM DOS (Jackson kernel)",
          color  = CLR_S, lw = 1.0,
          xlims  = XLIMS, ylims = (0, ρ_max * 1.12),
          legend = :topleft, grid = true, xlabel = "")
plot!(p1, E_ax, min.(ρ_analytic, ρ_max);
      label = "Analytic (broadened Van Hove)",
      color = :orange, lw = 1.1, ls = :dash, alpha = 0.8)
vline!(p1, band_edges; lw = 0.7, ls = :dot, color = CLR_ED,
       label = "sub-band edges  (G steps)")
annotate!(p1, 0.0, ρ_max*0.95,
    text("each dashed line = +1 quantum of conductance", 7, :center, :gray))

# Panel 2: Conductance staircase
p2 = plot(E_ax, G_exact;
          xlabel = "Fermi energy  E_F / t",
          ylabel = "G  (e²/h)",
          title  = "Landauer conductance  G = N_ch × e²/h",
          label  = "Exact  N_ch(E_F)  [open channel count]",
          color  = CLR_E, lw = 2.2, ls = :dash,
          xlims  = XLIMS, ylims = (-0.4, W + 0.5),
          legend = :topleft, grid = true)
plot!(p2, E_ax, Float64.(G_kpm);
      label  = "KPM  N_ch(E_F)  [mode-projected DOS threshold]",
      color  = CLR_KPM, lw = 1.6, alpha = 0.85)
hline!(p2, 0:W; lw = 0.4, ls = :dot, color = :gray, label = nothing)
vline!(p2, band_edges; lw = 0.7, ls = :dot, color = CLR_ED, label = nothing)

for nc in 1:W
    idx = findall(G_exact .== nc);  isempty(idx) && continue
    run_end = idx[1]
    for k in 2:length(idx); idx[k]==idx[k-1]+1 ? (run_end=idx[k]) : break; end
    Emid = E_ax[div(idx[1]+run_end, 2)]
    annotate!(p2, Emid, nc + 0.38, text("$nc", 7, :center, :navy))
end

p = plot(p1, p2; layout = (2, 1), size = (920, 680),
         left_margin = 5Plots.mm, bottom_margin = 3Plots.mm)
display(p)
outfile = joinpath(@__DIR__, "quantized_conductance_kpm.png")
savefig(p, outfile)
println("Saved: $outfile")

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
println("\n══ Summary ══════════════════════════════════════════════════════")
println("  KPM (N=$N sites, M=$N_mom moments) resolves $W sub-band edges.")
println("  Each KPM Van Hove peak marks a conductance step of e²/h.")
println("  G = N_ch × e²/h is directly readable from the KPM spectrum.")
println("  Exact sub-band edge energies: ", round.(sort(band_edges); digits=3))
println("══════════════════════════════════════════════════════════════════")
