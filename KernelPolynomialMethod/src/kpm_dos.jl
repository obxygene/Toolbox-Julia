# ============================================================
# kpm_dos.jl  –  High-level DOS pipeline
# ============================================================

"""
    kpm_dos(H; N_randvec=20, N_moments=200, N_points=nothing,
            kernel="Jackson", epsilon=0.05, bounds=nothing)
    -> (omega, DOS)

Compute the density of states  ρ(E) = (1/N) Tr[δ(E-H)]  without explicit
diagonalisation. Full pipeline:

  1. `kpm_scale_hamiltonian`  – rescale spectrum to (-1,1)
  2. `kpm_moments_spectrum`   – stochastic Chebyshev moments
  3. `kpm_kernel_correction`  – Jackson kernel damping
  4. `kpm_dct`                – DCT spectral reconstruction
  5. `kpm_rescale`            – map abscissas to physical energies

# Arguments
- `H`         : Sparse/dense Hermitian Hamiltonian.
- `N_randvec` : Number of stochastic random vectors (~20 typical).
- `N_moments` : Number of Chebyshev moments.
- `N_points`  : Output grid size (default: `2*N_moments`).
- `kernel`    : `"Jackson"` (default) or `"Lorentz"`.
- `epsilon`   : Spectral margin for scaling.
- `bounds`    : Optional `(E_min, E_max)` to skip eigenvalue computation.

# Returns
`(omega, DOS)` — physical energy axis and DOS normalised per site.

Matches MATLAB `KPM_DOS`.
"""
function kpm_dos(H;
                 N_randvec::Int    = 20,
                 N_moments::Int    = 200,
                 N_points::Union{Int,Nothing} = nothing,
                 kernel::String    = "Jackson",
                 epsilon::Real     = 0.05,
                 bounds            = nothing)

    Np = (N_points === nothing) ? 2 * N_moments : N_points

    # Step 1: rescale
    H_tilde, a, b = kpm_scale_hamiltonian(H; epsilon=epsilon, bounds=bounds)

    # Step 2: stochastic moments
    moments = kpm_moments_spectrum(H_tilde, N_randvec, N_moments)

    # Step 3: kernel damping
    moments_k = kpm_kernel_correction(moments, N_moments; kernel=kernel)

    # Step 4: DCT reconstruction
    omega_dct, DOS = kpm_dct(moments_k, Np)

    # Normalisation: √N_moments factor (matches MATLAB KPM_DOS line 55)
    DOS .*= sqrt(N_moments)

    # Step 5: physical energy axis
    omega = kpm_rescale(omega_dct, a, b)

    return omega, DOS
end
