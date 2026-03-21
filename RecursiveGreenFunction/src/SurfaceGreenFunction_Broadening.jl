"""
    surface_green_function_broadening(Sigma)

Compute the level-width (broadening) matrix Γ = i(Σ - Σ†).

# Physics Background
Γ is the anti-Hermitian part of the self-energy (times i). It enters
the Fisher-Lee transmission formula: T = Tr[Γ_L G^ret Γ_R G^adv].

# Arguments
- `Sigma::AbstractMatrix`: Retarded self-energy matrix.

# Returns
- `Gamma::Matrix{Float64}`: Level-width matrix Γ = i(Σ - Σ†).
"""
function surface_green_function_broadening(Sigma::AbstractMatrix)
    return real.(1im .* (Sigma .- Sigma'))
end
