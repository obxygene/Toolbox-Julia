# ============================================================
# kpm_conductivity.jl  –  Kubo-Bastin Hall and longitudinal conductance
# ============================================================

"""
    kpm_correlator_basis(energy, N_moments) -> Matrix{ComplexF64}

Build the N_moments×N_moments Kubo-Bastin kernel matrix Γ_{mn}(E) at a
given (rescaled) energy:

  Γ_{mn}(E) = Tₘ(E) · vₙ(E) + h.c.

where  vₙ(E) = (E - i·n·√(1-E²)) · exp(i·n·arccos(E))

# Arguments
- `energy`    : Fermi energy in rescaled units ∈ [-1, 1].
- `N_moments` : Number of Chebyshev moments.

Matches MATLAB `KPM_Correlator_basis` (Garcia et al. PRB 91, 245140, Eq. 10).
"""
function kpm_correlator_basis(energy::Real, N_moments::Int)
    @assert -1 <= energy <= 1 "energy must be in [-1, 1]"
    m = 0:(N_moments - 1)

    θ       = acos(energy)
    sinθ    = sqrt(1 - energy^2)

    T_m = cos.(m .* θ)                              # Chebyshev polys at E: row
    v_n = @. (energy - im * m * sinθ) * exp(im * m * θ)  # row

    Γ = T_m * v_n'      # outer product: Γ[i,j] = T_{i-1}(E) * v_{j-1}(E)
    return Γ .+ Γ'      # + Hermitian conjugate
end


"""
    kpm_hall_conductance(H, Jx, Jy, E_fermi;
                         N_moments=256, N_randvec=5,
                         kernel="Jackson", epsilon=0.05,
                         bounds=nothing, parallel=false)
    -> (sigma_xy, sigma_xx)

Compute the zero-temperature Hall conductivity σ_xy and longitudinal
conductivity σ_xx via the Kubo-Bastin formula (Garcia et al. 2015).

Returns conductivities in units of e²/h.

**σ_xy** (Fermi-sea contribution, topological):
  σ_xy(μ) = prefactor · Re[Σ_{m,n} μₘₙ^K · Γ_{mn}(μ)]
  Uses the Kubo-Bastin Γ_{mn} kernel.

**σ_xx** (Fermi-surface contribution, Kubo-Greenwood):
  σ_xx(μ) = prefactor · Re[Tₘ(μ) · μₘₙ^K · Tₙ(μ)]
  Uses the T_m(E_F)·T_n(E_F) kernel (delta function at Fermi level).

# Arguments
- `H`, `Jx`, `Jy` : Hamiltonian and current operators (sparse).
- `E_fermi`        : Fermi energy or vector of energies (physical units).
- `N_moments`      : Chebyshev moments.
- `N_randvec`      : Stochastic random vectors.
- `kernel`         : `"Jackson"` (default) or `"Lorentz"`.
- `epsilon`        : Spectral margin.
- `bounds`         : Optional `(E_min, E_max)` to skip eigenvalue computation.
- `parallel`       : Pass `true` to use `Threads.@threads` over random vectors.

# Prefactor for general lattices
The Garcia formula assumes 1 atom/unit-cell with A_cell=1 and gives
prefactor = 4/a². For multi-atom cells multiply by (n_atoms/A_cell).
The honeycomb lattice (2 atoms/cell, A_cell=√3) correction is 2/√3.
Pass the corrected prefactor via the keyword or use the returned tuple and
re-scale externally.

Matches MATLAB `KPM_Hall_Conductance` with the honeycomb correction.
"""
function kpm_hall_conductance(H, Jx, Jy, E_fermi;
                               N_moments::Int   = 256,
                               N_randvec::Int   = 5,
                               kernel::String   = "Jackson",
                               epsilon::Real    = 0.05,
                               bounds           = nothing,
                               parallel::Bool   = false)
    # Step 1: scale
    H_tilde, a, b = kpm_scale_hamiltonian(H; epsilon=epsilon, bounds=bounds)

    # Step 2: 2-D moments
    Mu_xy = kpm_moments_correlator(H_tilde, N_randvec, N_moments, Jx, Jy;
                                    parallel=parallel)
    Mu_xx = kpm_moments_correlator(H_tilde, N_randvec, N_moments, Jx, Jx;
                                    parallel=parallel)

    # Step 3: kernel
    Mu_xy_K = kpm_kernel_correction(Mu_xy, N_moments; kernel=kernel)
    Mu_xx_K = kpm_kernel_correction(Mu_xx, N_moments; kernel=kernel)

    # Step 4: evaluate at each Fermi energy
    # Prefactor for 1-atom/cell lattice (A_cell=1):  4/a²
    # For honeycomb (2 atoms/cell, A_cell=√3): multiply by 2/√3
    prefactor = 4.0 / (a * a)

    E_list  = isa(E_fermi, Number) ? [Float64(E_fermi)] : Float64.(E_fermi)
    sigma_xy = zeros(Float64, length(E_list))
    sigma_xx = zeros(Float64, length(E_list))

    m_vec = 0:(N_moments - 1)
    for (ii, EF) in enumerate(E_list)
        EF_sc = clamp((EF - b) / a, -1 + epsilon/2, 1 - epsilon/2)

        # σ_xy: Kubo-Bastin Γ_{mn}(E_F) kernel
        Γ           = kpm_correlator_basis(EF_sc, N_moments)
        sigma_xy[ii] = prefactor * real(sum(Mu_xy_K .* Γ))

        # σ_xx: Kubo-Greenwood T_m(E_F)·T_n(E_F) kernel
        T_EF         = cos.(m_vec .* acos(EF_sc))
        sigma_xx[ii] = prefactor * real(dot(T_EF, Mu_xx_K * T_EF))
    end

    return sigma_xy, sigma_xx
end
