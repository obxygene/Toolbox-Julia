# ============================================================
# kpm_kernel.jl  –  Chebyshev moment damping kernels
# ============================================================

"""
    kpm_kernel(N_moments; kernel="Jackson", lambda=3) -> Vector{Float64}

Return `N_moments` damping coefficients {gₙ} that suppress Gibbs oscillations
in the reconstructed spectral function.

## Kernels

**Jackson** (default, minimises mean-square truncation error, Weisse Eq. 71):
  gₙ = [(N-n+1)·cos(πn/(N+1)) + sin(πn/(N+1))·cot(π/(N+1))] / (N+1)

**Lorentz** (smoother but broader features, Weisse Eq. 76):
  gₙ = sinh(λ·(1 - n/N)) / sinh(λ)

# Arguments
- `N_moments` : Number of Chebyshev moments.
- `kernel`    : `"Jackson"` or `"Lorentz"`.
- `lambda`    : Lorentz parameter (default 3; larger → broader features).
"""
function kpm_kernel(N_moments::Int; kernel::String = "Jackson", lambda::Real = 3)
    n = 0:(N_moments - 1)

    if kernel == "Jackson"
        d = N_moments + 1
        g = @. ((N_moments + 1 - n) * cos(π * n / d) +
                sin(π * n / d) * cot(π / d)) / d
    elseif kernel == "Lorentz"
        g = @. sinh(lambda * (1 - n / N_moments)) / sinh(lambda)
    else
        error("Unknown kernel \"$kernel\". Choose \"Jackson\" or \"Lorentz\".")
    end
    return g
end


"""
    kpm_kernel_correction(moments, N_moments; kernel="Jackson", lambda=3)

Apply kernel damping to raw Chebyshev moments.

- 1-D (vector): μₙᴷ = gₙ · μₙ
- 2-D (matrix): μₘₙᴷ = gₘ · gₙ · μₘₙ  (with g₀ halved for DCT convention)

Matches MATLAB `KPM_Kernel_Correction`.
"""
function kpm_kernel_correction(moments::AbstractVecOrMat,
                                N_moments::Int;
                                kernel::String = "Jackson",
                                lambda::Real   = 3)
    g = kpm_kernel(N_moments; kernel=kernel, lambda=lambda)

    if ndims(moments) == 1 || size(moments, 1) == 1 || size(moments, 2) == 1
        # 1-D case
        return g .* vec(moments)
    else
        # 2-D case: outer-product kernel; halve g₀ for DCT normalisation
        gc = copy(g)
        gc[1] /= 2
        return (gc .* gc') .* moments   # element-wise outer product
    end
end
