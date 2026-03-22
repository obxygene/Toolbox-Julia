"""
    surface_green_function_v2(H00, V, omega, accuracy; eta=1e-4)

Surface Green function via convergence-controlled Lopez-Sancho iteration.

Same algorithm as `surface_green_function` but terminates when
the change in the surface self-energy Σ drops below `accuracy`,
rather than running a fixed number of steps.

# Arguments
- `H00::AbstractMatrix`: (W×W) Intra-layer Hamiltonian.
- `V::AbstractMatrix`: (W×W) Forward inter-layer hopping matrix.
- `omega::Number`: Energy at which G₀₀ is evaluated.
- `accuracy::Real`: Convergence threshold on ‖Σ‖_F change.
- `eta::Real=1e-4`: Infinitesimal broadening.

# Returns
- `G_00::Matrix{ComplexF64}`: Retarded surface Green function.
- `count::Int`: Number of iterations performed.

# References
M.P. Lopez Sancho, J.M. Lopez Sancho, J. Rubio,
J. Phys. F: Met. Phys. 15, 851 (1985).
"""
function surface_green_function_v2(H00::AbstractMatrix, V::AbstractMatrix,
                                   omega::Number, accuracy::Real;
                                   eta::Real=1e-4)
    W = size(H00, 1)
    Iden = Matrix{ComplexF64}(I, W, W)

    alpha       = ComplexF64.(V)
    beta        = ComplexF64.(V')
    Sigma       = ComplexF64.(H00)
    Sigma_tilde = ComplexF64.(H00)

    count = 0
    err = Inf

    while abs(err) > accuracy
        Sigma_old = copy(Sigma)

        g_temp = ((omega + im*eta) * Iden - Sigma_tilde) \ Iden

        Sigma       .= Sigma       .+ alpha * g_temp * beta
        Sigma_tilde .= Sigma_tilde .+ beta * g_temp * alpha .+ alpha * g_temp * beta

        alpha = alpha * g_temp * alpha
        beta  = beta  * g_temp * beta

        err = norm(Sigma - Sigma_old)
        count += 1
    end

    G_00 = ((omega + im*eta) * Iden - Sigma) \ Iden
    return G_00, count
end
