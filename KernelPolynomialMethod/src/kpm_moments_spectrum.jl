# ============================================================
# kpm_moments_spectrum.jl  –  1-D Chebyshev moments for DOS
# ============================================================

"""
    kpm_moments_spectrum(H_scaled, N_randvec, N_moments) -> Vector{Float64}

Compute Chebyshev moments μₙ = (1/N) Tr[Tₙ(H)] for the spectral density via
the stochastic trace estimator:

  μₙ ≈ (1/R) Σᵣ ⟨r| Tₙ(H) |r⟩

Uses the **doubled-index trick** (Weisse et al., Sec. IV) which halves the
number of sparse matrix-vector products: μ_{2k-1} and μ_{2k} are computed
simultaneously from the inner products ⟨α|α⟩ and ⟨α₊|α⟩.

# Arguments
- `H_scaled`   : Hamiltonian rescaled to [-1,1] (sparse recommended).
- `N_randvec`  : Number of stochastic random phase vectors R (~20 typical).
- `N_moments`  : Total number of Chebyshev moments (must be even or odd ≥ 2).

# Returns
Real vector of length `N_moments`, averaged over `N_randvec` random vectors.
Matches MATLAB `KPM_Moments_Spectrum`.
"""
function kpm_moments_spectrum(H_scaled, N_randvec::Int, N_moments::Int)
    N = size(H_scaled, 1)
    moments = zeros(Float64, N_moments)

    for _ in 1:N_randvec
        # Random complex phase vector (uniform on unit sphere; reduces bias vs real)
        r = exp.(2π * im .* rand(N))
        r ./= norm(r)

        α_minus = r
        α       = H_scaled * α_minus

        m_temp       = zeros(ComplexF64, N_moments)
        m_temp[1]    = dot(r, r)          # μ₀ = 1 (normalised vector)
        m_temp[2]    = dot(r, α)          # μ₁

        # Doubled-index trick
        for k in 2:(N_moments ÷ 2)
            α_plus      = 2 .* (H_scaled * α) .- α_minus
            m_temp[2k-1] = 2 * dot(α,      α)       - m_temp[1]
            m_temp[2k]   = 2 * dot(α_plus, α)       - m_temp[2]
            α_minus = α
            α       = α_plus
        end
        # Handle odd N_moments
        if isodd(N_moments)
            α_plus = 2 .* (H_scaled * α) .- α_minus
            m_temp[end] = 2 * dot(α_plus, α_plus) - m_temp[1]
        end

        moments .+= real.(m_temp)
    end

    return moments ./ N_randvec
end
