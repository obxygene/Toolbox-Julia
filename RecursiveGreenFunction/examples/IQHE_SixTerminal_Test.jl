"""
IQHE_SixTerminal_Test.jl
=========================================================================
Integer Quantum Hall Effect — Six-Terminal Hall Bar
Correct implementation using narrow probes at single x-layers.

KEY FIX vs previous version
  Previous: probe terminals spanned W=20 sites across 20 different x-layers.
  The W×W off-diagonal self-energy then created unphysical long-range
  coupling between layers, short-circuiting the edge states (T[2,3]≈70).

  This version: probes are W_probe-wide and attach at a SINGLE x-layer each.
  Off-diagonal Sigma couples y-sites within the same layer only — physical.
  The probe surface GF is computed with the correct Peierls phases for the
  probe's actual y-positions (top or bottom edge).  Matches MATLAB
  SixTerminal_RGF_Benchmark.m exactly.

GEOMETRY  (Landau gauge A = (B·y, 0, 0))
  Lead 1 = left  (source, full width W)
  Lead 4 = right (drain,  full width W)
  Lead 2 = top-left  probe  (W_probe sites at x=x_probe_L, y=W-Wp+1..W)
  Lead 3 = top-right probe  (W_probe sites at x=x_probe_R, y=W-Wp+1..W)
  Lead 5 = bot-right probe  (W_probe sites at x=x_probe_R, y=1..W_probe)
  Lead 6 = bot-left  probe  (W_probe sites at x=x_probe_L, y=1..W_probe)

ALGORITHM
  Sparse partial GF: build A = (ω+iη)I − H_CC − Σ_embedded (sparse),
  solve A \ [e_{T1} … e_{T6}] in one batch (single LU), extract W-wide
  sub-blocks, apply Fisher–Lee T(j,k)=Tr[Γ_j G_{jk} Γ_k G_{jk}†].
=========================================================================
"""

using LinearAlgebra
using SparseArrays
using Printf

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()
using RecursiveGreenFunction

# ====================================================================
#  PARAMETERS
# ====================================================================
const omega    = -3.5      # Fermi energy  [units of t]
const Width    = 20        # strip width W  (y-direction)
const Length   = 60        # strip length L  (x-direction layers)
const t        = 1.0       # hopping amplitude
const eta      = 1e-6      # infinitesimal broadening
const Disorder = 0.0       # on-site disorder (0 = clean)

# Probe geometry
# W_probe ≥ 4 is needed for propagating modes at ω = -3.5
# (band bottom of a W_probe-wide strip ≈ -2cos(π/(Wp+1)) - 2;
#  for Wp=4 this is ≈ -3.618, so ω = -3.5 is inside the band)
const W_probe   = 5
const x_probe_L = 20       # layer where left  probes (2,6) attach
const x_probe_R = 40       # layer where right probes (3,5) attach

const N = Width * Length

phi_list = range(0.0, 0.25, length=101)
N_phi    = length(phi_list)

# ====================================================================
#  TERMINAL LABELS  (site index = (x-1)*W + y, 1-indexed)
# ====================================================================
lbl(x, y) = (x - 1)*Width + y

ys_top = (Width - W_probe + 1):Width     # top probe y-positions
ys_bot = 1:W_probe                        # bottom probe y-positions

Terminal = Vector{Vector{Int}}(undef, 6)
Terminal[1] = collect(1:Width)                                   # left lead
Terminal[4] = collect((N - Width + 1):N)                         # right lead
Terminal[2] = [lbl(x_probe_L, y) for y in ys_top]               # top-left probe
Terminal[3] = [lbl(x_probe_R, y) for y in ys_top]               # top-right probe
Terminal[5] = [lbl(x_probe_R, y) for y in ys_bot]               # bot-right probe
Terminal[6] = [lbl(x_probe_L, y) for y in ys_bot]               # bot-left probe

@printf("W=%d  L=%d  W_probe=%d  x_probe_L=%d  x_probe_R=%d\n",
        Width, Length, W_probe, x_probe_L, x_probe_R)
println("Terminal sizes: ", [length(Terminal[k]) for k in 1:6])

col_off = cumsum([0; [length(Terminal[k]) for k in 1:6]])

# ====================================================================
#  BAND STRUCTURE — HOFSTADTER BUTTERFLY
# ====================================================================
println("\n── Hofstadter butterfly (collecting eigenvalues) ──")
H_00 = -t .* (diagm(1 => ones(Width-1)) .+ diagm(-1 => ones(Width-1)))
H_00_p = -t .* (diagm(1 => ones(W_probe-1)) .+ diagm(-1 => ones(W_probe-1)))

kx_bs   = range(0, 2π, length=80)
kx_fine = range(0, 2π, length=200)

bfly_phi = Float64[]
bfly_E   = Float64[]
for phi_b in phi_list
    H01_b = -t .* Diagonal(exp.(im .* (1:Width) .* phi_b))
    for kx in kx_bs
        Hk = H_00 .+ exp(im*kx) .* H01_b .+ exp(-im*kx) .* H01_b'
        append!(bfly_E,   real.(eigvals(Hermitian(Hk))))
        append!(bfly_phi, fill(phi_b, Width))
    end
end

phi_sel = [0.05, 0.10, 0.15, 0.20, 0.25]
disp_kx = collect(kx_fine)
disp_E  = [zeros(length(kx_fine), Width) for _ in phi_sel]
for (ip, phi_b) in enumerate(phi_sel)
    H01_b = -t .* Diagonal(exp.(im .* (1:Width) .* phi_b))
    for (ik, kx) in enumerate(kx_fine)
        Hk = H_00 .+ exp(im*kx) .* H01_b .+ exp(-im*kx) .* H01_b'
        disp_E[ip][ik, :] .= sort(real.(eigvals(Hermitian(Hk))))
    end
end
println("Band structure data collected.")

try
    @eval using Plots
    phi_arr = collect(phi_list)

    p1 = scatter(bfly_phi, bfly_E,
                 markersize=1, markerstrokewidth=0, markercolor=:royalblue,
                 xlabel="φ  (flux per plaquette)",
                 ylabel="Energy  [t]",
                 title="Hofstadter butterfly  (W=$Width)",
                 legend=false, ylims=(-4.5, 4.5))
    hline!(p1, [omega], color=:red, lw=2, label="ω=$omega")

    p2 = plot(xlabel="k_x", ylabel="Energy  [t]",
              title="E(kx) at selected φ", ylims=(-4.5, 4.5))
    for (ip, phi_b) in enumerate(phi_sel)
        plot!(p2, disp_kx, disp_E[ip], color=ip, lw=0.8,
              label="φ=$(phi_b)")
    end
    hline!(p2, [omega], color=:red, lw=2, ls=:dash, label="ω=$omega")

    fig_bs = plot(p1, p2, layout=(1,2), size=(1200, 460),
                  plot_title="Band structure: is ω=$omega in a gap?")
    display(fig_bs)
    savefig(fig_bs, "IQHE_BandStructure.png")
    println("Band structure saved → IQHE_BandStructure.png")
catch e
    println("(Plots.jl unavailable for band structure: $e)")
    println("Nearest band energy to ω=$omega at each selected phi:")
    for phi_b in phi_sel
        mask = abs.(bfly_phi .- phi_b) .< 1e-8
        nearest = minimum(abs.(bfly_E[mask] .- omega))
        @printf("  φ=%.2f: min|E-ω| = %.4f\n", phi_b, nearest)
    end
end
println()

# ====================================================================
#  MAIN SWEEP
# ====================================================================
R_H = zeros(N_phi)
R_L = zeros(N_phi)

@printf("Running IQHE sweep: W=%d  L=%d  N=%d  ω=%.2f  Wp=%d\n",
        Width, Length, N, omega, W_probe)
t_start = time()

for ii in 1:N_phi
    phi = phi_list[ii]

    # ── Layer Hamiltonians ──────────────────────────────────────────
    H_01 = -t .* Diagonal(exp.(im .* (1:Width) .* phi))

    # ── Surface GF for longitudinal leads (full width W) ────────────
    G_surf_L, = surface_green_function_v2(H_00, Matrix(H_01), omega, 1e-10; eta=eta)
    Sigma_L   = surface_green_function_self_energy(G_surf_L, Matrix(H_01))
    Sigma_R   = transpose(Sigma_L)       # non-conjugate transpose (Landau gauge)

    # ── Surface GF for narrow probes (W_probe wide) ─────────────────
    # Top probe: y-positions ys_top  (physically at y = Width-Wp+1 .. Width)
    H_01_top     = -t .* Diagonal(exp.(im .* collect(ys_top) .* phi))
    G_surf_top,  = surface_green_function_v2(H_00_p, Matrix(H_01_top), omega, 1e-10; eta=eta)
    Sigma_top    = surface_green_function_self_energy(G_surf_top, Matrix(H_01_top))

    # Bottom probe: y-positions ys_bot  (physically at y = 1..Wp)
    H_01_bot     = -t .* Diagonal(exp.(im .* collect(ys_bot) .* phi))
    G_surf_bot,  = surface_green_function_v2(H_00_p, Matrix(H_01_bot), omega, 1e-10; eta=eta)
    Sigma_bot    = surface_green_function_self_energy(G_surf_bot, Matrix(H_01_bot))

    # Full-size (W×W) self-energy blocks for probe terminals
    # (embed W_probe×W_probe into W×W at the probe y-rows)
    Sigma_TL = zeros(ComplexF64, Width, Width)
    Sigma_TR = zeros(ComplexF64, Width, Width)
    Sigma_BL = zeros(ComplexF64, Width, Width)
    Sigma_BR = zeros(ComplexF64, Width, Width)
    Sigma_TL[ys_top, ys_top] .= Sigma_top
    Sigma_TR[ys_top, ys_top] .= Sigma_top
    Sigma_BL[ys_bot, ys_bot] .= Sigma_bot
    Sigma_BR[ys_bot, ys_bot] .= Sigma_bot

    # Ordered to match Terminal numbering 1…6
    Sigma = [Sigma_L, Sigma_TL, Sigma_TR, Sigma_R, Sigma_BR, Sigma_BL]

    # Small broadening matrices  Γ = i(Σ − Σ†)  at each terminal
    Gamma_small = [real.(im .* (S .- S')) for S in Sigma]

    # ── Sparse central Hamiltonian ───────────────────────────────────
    H_CC = kron(sparse(I, Length, Length), sparse(complex.(H_00))) .+
           kron(sparse(diagm(1 => ones(Length-1))), sparse(H_01)) .+
           kron(sparse(diagm(-1 => ones(Length-1))), sparse(H_01'))

    if Disorder > 0.0
        H_CC += Diagonal(Disorder .* (rand(N) .- 0.5))
    end

    # ── Build  A = (ω+iη)I − H_CC − Σ_embedded (sparse) ───────────
    # Probe self-energies embed at SINGLE x-layers (no cross-layer coupling):
    #   Terminal[2] sites: (x_probe_L, ys_top) → same x, same layer
    #   Off-diagonal Sigma_TL connects y-sites within that one layer only.
    A = (omega + im*eta) .* sparse(I, N, N) .- H_CC
    dropzeros!(A)

    for jj in 1:6
        idx = Terminal[jj]
        A[idx, idx] .-= Sigma[jj][get_rows(jj), get_rows(jj)]
    end

    # ── Partial GF: one sparse solve for all terminal columns ────────
    n_rhs = col_off[end]
    RHS   = zeros(ComplexF64, N, n_rhs)
    for kk in 1:6
        for (q, s) in enumerate(Terminal[kk])
            RHS[s, col_off[kk] + q] = one(ComplexF64)
        end
    end
    G_cols = A \ RHS

    # ── Conductance matrix via Fisher–Lee ───────────────────────────
    #   T(j,k) = Tr[Γ_j_small · G_{jk} · Γ_k_small · G_{jk}†]
    #          = sum((Γ_j · G_{jk} · Γ_k) .* conj(G_{jk}))
    C = zeros(6, 6)
    for jj in 1:6
        rj = get_rows(jj)
        Gj = Gamma_small[jj][rj, rj]
        for kk in 1:6
            jj == kk && continue
            rk = get_rows(kk)
            Gk      = Gamma_small[kk][rk, rk]
            G_block = G_cols[Terminal[jj], col_off[kk]+1 : col_off[kk+1]]
            T_jk    = real(sum((Gj * G_block * Gk) .* conj.(G_block)))
            C[jj, kk] = -T_jk
        end
        C[jj, jj] = -sum(C[jj, :])
    end

    # ── Büttiker equations ────────────────────────────────────────────
    src = [1, 4];  prb = [2, 3, 5, 6]
    V_s  = [1.0; 0.0]
    C_ss = C[src, src];  C_sp = C[src, prb]
    C_ps = C[prb, src];  C_pp = C[prb, prb]

    V_p = -(C_pp \ C_ps) * V_s
    I_s = (C_ss - C_sp * (C_pp \ C_ps)) * V_s

    V_all       = zeros(6)
    V_all[src] .= V_s
    V_all[prb] .= V_p

    R_H[ii] = real(V_all[2] - V_all[6]) / real(I_s[1])
    R_L[ii] = real(V_all[2] - V_all[3]) / real(I_s[1])

    if mod(ii, 10) == 0
        @printf("  φ=%.4f  R_H=%+.4f  R_L=%+.6f  (%.1f s)\n",
                phi, R_H[ii], R_L[ii], time() - t_start)
    end
end
@printf("Total: %.1f s\n", time() - t_start)

# ====================================================================
#  PLOT RESULTS
# ====================================================================
try
    @eval using Plots
    phi_arr = collect(phi_list)

    p1 = plot(phi_arr, R_H, lw=2, color=:royalblue,
              xlabel="φ  (flux per plaquette, Φ₀=2π)",
              ylabel="R_H  [h/e²]",
              title="Hall Resistance  (W=$Width, L=$Length, ω=$omega, Wp=$W_probe)",
              ylims=(-0.1, 1.4), grid=true, label="R_H")
    for n in 1:5
        hline!(p1, [1/n], ls=:dash, lw=0.8, label="1/$n")
    end

    p2 = plot(phi_arr, R_L, lw=2, color=:red,
              xlabel="φ  (flux per plaquette, Φ₀=2π)",
              ylabel="R_L  [h/e²]",
              title="Longitudinal Resistance",
              grid=true, label="R_L")
    hline!(p2, [0.0], color=:black, ls=:dash, lw=1.2, label="R_L = 0")

    fig = plot(p1, p2, layout=(2,1), size=(900, 650),
               plot_title="IQHE Six-Terminal Hall Bar")
    display(fig)
    savefig(fig, "IQHE_SixTerminal.png")
    println("Saved → IQHE_SixTerminal.png")
catch e
    println("(Plots.jl unavailable: $e)")
    println("\nφ\t\tR_H\t\tR_L")
    for ii in 1:5:N_phi
        @printf("%.4f\t\t%+.4f\t\t%+.4f\n", phi_list[ii], R_H[ii], R_L[ii])
    end
end

# ====================================================================
#  HELPERS  (defined after use; Julia allows this in scripts)
# ====================================================================
function get_rows(jj)
    # Returns the y-row sub-indices within the W×W Sigma block
    # that are actually nonzero for terminal jj.
    if jj == 1 || jj == 4
        return 1:Width           # full-width longitudinal leads
    elseif jj == 2 || jj == 3   # top probes
        return ys_top
    else                         # bottom probes (5, 6)
        return ys_bot
    end
end
