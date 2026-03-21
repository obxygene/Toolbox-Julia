# ============================================================
# kpm_dct.jl  –  DCT-based spectral reconstruction
# ============================================================

"""
    kpm_dct(moments_kernel, N_points) -> (omega, rho)

Reconstruct a spectral function from kernel-corrected Chebyshev moments
using a Type-III DCT (MATLAB `dct(...,'Type',3)` convention).

- **1-D vector** (DOS): applies DCT and divides by the Chebyshev arclength
  weight  1/(π·√(1-x²)).
- **2-D matrix** (Kubo-Bastin correlator): applies sequential DCTs along
  rows and columns, returning the raw expansion (no weight applied).

# Arguments
- `moments_kernel` : Kernel-corrected moments (vector for DOS, matrix for correlator).
- `N_points`       : Number of output energy points (standard: 2×N_moments).

# Returns
`(omega, rho)` where `omega` contains the Chebyshev abscissas in `(-1,1)`
and `rho` is the reconstructed spectral function.
Convert `omega` to physical units with `kpm_rescale`.

Matches MATLAB `KPM_DCT`.

## Implementation note
MATLAB's `dct(x, N, 'Type', 3)` uses the normalisation
  y(k) = √(2/N) · [x(1)/√2 + Σₙ₌₂ᴺ x(n)·cos(π(2k-1)(n-1)/(2N))]
FFTW's `REDFT01` computes
  y[k] = x[0] + 2·Σₙ₌₁ᴺ⁻¹ x[n]·cos(π·n·(k+0.5)/N)
Matching requires rescaling the input: x_fftw = [μ₀/2; μ₁/2; …; μ_{M-1}/2]
and then y_matlab = √(2/N) · y_fftw.  The extra √(2/N) prefactor is absorbed
into the DOS normalisation in `kpm_dos`.
"""
function kpm_dct(moments_kernel, N_points::Int)
    omega = kpm_chebyshev_abscissas(N_points)

    if ndims(moments_kernel) == 1 || size(moments_kernel, 1) == 1 || size(moments_kernel, 2) == 1
        # --- 1-D DOS reconstruction ---
        m = vec(Float64.(moments_kernel))
        M = length(m)

        # FFTW REDFT01 computes:
        #   y[k] = x[0] + 2·Σ_{j=1}^{N-1} x[j]·cos(π·j·(2k+1)/(2N))
        # KPM sum at Chebyshev node k:
        #   S[k] = μ₀/2 + Σ_{j=1}^{N-1} μⱼ·cos(...)
        # Setting x[0]=μ₀/2, x[j]=μⱼ/2 for j≥1 gives y[k]=S[k].
        # MATLAB's DCT-III = √(2/N)·S, so multiply output by √(2/N).
        m ./= 2   # divide ALL moments by 2 (DC and AC alike)

        # Zero-pad to N_points
        x = vcat(m, zeros(Float64, N_points - M))

        # FFTW DCT-III → y[k] = μ₀/2 + Σ μⱼ·cos  (the KPM Chebyshev sum)
        y_raw = FFTW.r2r(x, FFTW.REDFT01)

        # Apply MATLAB normalisation √(2/N_points) so units match
        y = reverse(y_raw) .* sqrt(2 / N_points)

        # Divide by Chebyshev arclength weight  π·√(1-ω²)
        rho = y ./ (π .* sqrt.(1 .- omega .^ 2))

    else
        # --- 2-D correlator reconstruction ---
        # Apply DCT along rows, then columns (matching MATLAB's sequential dct calls)
        M1, M2 = size(moments_kernel)
        m = ComplexF64.(moments_kernel)
        # Same logic as 1-D: divide all by 2, then correct (m[0]*m[0] term
        # would be divided by 4 but each DCT axis only sees /2, so just do /2 per axis)
        m ./= 2   # first axis
        # Second DCT pass also needs /2 for DC; done by passing m./2 again below

        # Pad to N_points × N_points
        padded = zeros(ComplexF64, N_points, N_points)
        padded[1:M1, 1:M2] = m

        # DCT along rows (second index), then columns (first index).
        # Each 1-D pass: input already divided by 2 above; output scaled by √(2/N).
        # Second pass: the DC of the *row output* also needs /2 → divide col-input by 2.
        temp = mapslices(row -> reverse(FFTW.r2r(row ./ 2, FFTW.REDFT01)), padded, dims=2)
        temp .*= sqrt(2 / N_points)
        rho  = mapslices(col -> reverse(FFTW.r2r(real.(col), FFTW.REDFT01)), temp, dims=1)
        rho .*= sqrt(2 / N_points)
    end

    return omega, rho
end
