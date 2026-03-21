# ============================================================
# kpm_scale.jl  –  Hamiltonian rescaling and helper utilities
# ============================================================

"""
    kpm_scale_hamiltonian(H; epsilon=0.05, bounds=nothing) -> (H_tilde, a, b)

Rescale a Hermitian matrix `H` so that its spectrum lies in `(-1+ε/2, 1-ε/2)`.

Applies the affine map  H̃ = (H - b·I) / a  where
  a = (E_max - E_min) / (2 - ε)   (half-bandwidth after scaling)
  b = (E_max + E_min) / 2          (spectral centre)

# Arguments
- `H`       : Sparse/dense Hermitian matrix.
- `epsilon` : Spectral margin to keep Chebyshev polynomials inside [-1,1].
- `bounds`  : Optional `(E_min, E_max)` tuple to skip the Arpack eigenvalue
              computation (useful when the spectrum is known analytically).

# Returns `(H_tilde, a, b)` matching MATLAB `KPM_scaleHamiltonian`.
"""
function kpm_scale_hamiltonian(H::AbstractMatrix{<:Number};
                                epsilon::Real = 0.05,
                                bounds::Union{Nothing, Tuple{Real,Real}} = nothing)
    if bounds !== nothing
        E_min, E_max = Float64(bounds[1]), Float64(bounds[2])
    else
        # Arpack sparse Lanczos (equivalent to MATLAB eigs(...,'largestreal'))
        λ_max_vec, = eigs(H, nev=1, which=:LR, tol=epsilon/2)
        λ_min_vec, = eigs(H, nev=1, which=:SR, tol=epsilon/2)
        E_max = real(λ_max_vec[1])
        E_min = real(λ_min_vec[1])
    end

    span = E_max - E_min
    @assert span > 0 "Hamiltonian has a single eigenvalue; spectrum is undefined."

    a = span / (2 - epsilon)
    b = (E_max + E_min) / 2

    # Use UniformScaling `b * I` (not `b * I(N)` which creates a dense Diagonal)
    # so the result stays sparse when H is sparse.
    H_tilde = (H - b * I) / a
    return H_tilde, a, b
end


"""
    kpm_chebyshev_abscissas(N_points) -> Vector{Float64}

Return `N_points` Chebyshev nodes of the first kind in ascending order:
  x_k = cos(π·(k - 0.5) / N_points),   k = 1 … N_points

These coincide with the Type-III DCT sample positions (MATLAB convention).
"""
function kpm_chebyshev_abscissas(N_points::Int)
    # Descending cosine argument gives ascending nodes
    ks = (N_points-1):-1:0          # 0-indexed descending
    return cos.(π .* (ks .+ 0.5) ./ N_points)
end


"""
    kpm_rescale(omega_dct, a, b) -> Array

Invert the Hamiltonian scaling:  ω = a·ω̃ + b
"""
function kpm_rescale(omega_dct, a::Real, b::Real)
    return a .* omega_dct .+ b
end


"""
    kpm_fermi_distribution(energy, mu, T) -> Array

Fermi-Dirac occupation  f(E) = 1 / (1 + exp((E - μ) / T)).
Returns the exact T=0 Heaviside step when `T == 0`.
"""
function kpm_fermi_distribution(energy, mu::Real, T::Real)
    @assert T >= 0 "Temperature must be non-negative."
    if T == 0
        return Float64.(energy .< mu)
    else
        return @. 1 / (1 + exp((energy - mu) / T))
    end
end
