# ============================================================
# kpm_moments_operator.jl  –  Operator-weighted Chebyshev moments
# ============================================================

"""
    kpm_moments_operator(H_scaled, N_randvec, N_moments, O) -> Vector{Float64}

Compute μₙ = (1/N) Tr[O · Tₙ(H)] via the stochastic estimator:

  μₙ ≈ (1/R) Σᵣ ⟨r|O · Tₙ(H)|r⟩

Unlike `kpm_moments_spectrum`, an arbitrary operator `O` is inserted.
Uses the standard Chebyshev recursion (O(N_moments) matvecs).

Matches MATLAB `KPM_Moments_Operator`.
"""
function kpm_moments_operator(H_scaled, N_randvec::Int, N_moments::Int,
                               O::AbstractMatrix)
    N = size(H_scaled, 1)
    moments    = zeros(ComplexF64, N_moments)
    moments[1] = 1.0    # μ₀ = Tr[O·I]/N ≈ 1 by convention

    for _ in 1:N_randvec
        r = exp.(2π * im .* rand(N))
        r ./= norm(r)

        α_minus = r
        α       = H_scaled * α_minus
        β       = r' * O          # row vector; avoids recomputing bra each step

        for k in 2:N_moments
            moments[k] += dot(β, α)          # β * α = ⟨r|O|Tₖ(H)|r⟩
            α_plus  = 2 .* (H_scaled * α) .- α_minus
            α_minus = α
            α       = α_plus
        end
    end

    # μ₀ is set analytically; average the rest
    result    = zeros(Float64, N_moments)
    result[1] = 1.0
    result[2:end] = real.(moments[2:end]) ./ N_randvec
    return result
end
