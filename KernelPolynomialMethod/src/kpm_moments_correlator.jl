# ============================================================
# kpm_moments_correlator.jl  –  2-D Chebyshev moments for Kubo-Bastin
# ============================================================

"""
    kpm_moments_correlator(H_scaled, N_randvec, N_moments, Oa, Ob;
                           parallel=false) -> Matrix{ComplexF64}

Compute 2-D Chebyshev moments for the Kubo-Bastin conductivity formula:

  μₘₙ = (1/N) Tr[Tₘ(H) Oₐ† · Tₙ(H) Ob]

using the stochastic trace estimator:

  μₘₙ ≈ (1/R) Σᵣ  Bras[:,m]' * Kets[:,n]

where  Kets[:,n] = Ob·Tₙ(H)|r⟩  and  Bras[:,m] = Oₐ·Tₘ(H)|r⟩.

The full N_moments×N_moments complex matrix is stored; the imaginary part is
needed for the Hall conductivity.

# Arguments
- `H_scaled`  : Hamiltonian rescaled to [-1,1].
- `N_randvec` : Number of random phase vectors R.
- `N_moments` : Number of Chebyshev moments per dimension.
- `Oa`, `Ob`  : Velocity/current operators (same size as H).
- `parallel`  : If `true`, use `Threads.@threads` over random vectors.
                Launch Julia with `julia --threads=N` for N cores.

# Memory note
Each thread allocates two N×M complex matrices (Kets_loc, Bras_loc).
Memory per thread ≈ 2 × N × M × 16 bytes.
Example: N=131072, M=1024 → ~4 GB per thread. Limit `--threads` accordingly.

Matches MATLAB `KPM_Moments_Correlator` (with "parallel" flag).
"""
function kpm_moments_correlator(H_scaled, N_randvec::Int, N_moments::Int,
                                 Oa::AbstractMatrix, Ob::AbstractMatrix;
                                 parallel::Bool = false)
    N = size(H_scaled, 1)

    if parallel
        return _correlator_parallel(H_scaled, N, N_randvec, N_moments, Oa, Ob)
    else
        return _correlator_serial(H_scaled, N, N_randvec, N_moments, Oa, Ob)
    end
end


# ------------------------------------------------------------------
# Serial path  (exact translation of the MATLAB serial loop)
# ------------------------------------------------------------------
function _correlator_serial(H_scaled, N::Int, N_randvec::Int, N_moments::Int,
                              Oa::AbstractMatrix, Ob::AbstractMatrix)
    Kets        = zeros(ComplexF64, N, N_moments)
    Bras        = zeros(ComplexF64, N, N_moments)
    moments_mat = zeros(ComplexF64, N_moments, N_moments)

    for _ in 1:N_randvec
        r = exp.(2π * im .* rand(N))
        r ./= norm(r)

        # Build Kets: Tₙ(H)|r⟩  for  n = 0 … N_moments-1
        α_minus      = r
        α            = H_scaled * α_minus
        Kets[:,1]   .= α_minus        # T₀|r⟩ = |r⟩
        Kets[:,2]   .= α              # T₁|r⟩ = H|r⟩
        for m in 3:N_moments
            α_plus     = 2 .* (H_scaled * α) .- α_minus
            Kets[:,m] .= α_plus
            α_minus, α = α, α_plus
        end
        # Apply Ob to all Ket columns at once (one BLAS dgemm)
        mul!(Kets, Ob, copy(Kets))

        # Build Bras: Tₘ(H)·Oₐ|r⟩  for  m = 0 … N_moments-1
        β_minus      = Oa * r
        β            = H_scaled * β_minus
        Bras[:,1]   .= β_minus
        Bras[:,2]   .= β
        for n in 3:N_moments
            β_plus     = 2 .* (H_scaled * β) .- β_minus
            Bras[:,n] .= β_plus
            β_minus, β = β, β_plus
        end

        # Accumulate: μₘₙ += ⟨Bras_m|Kets_n⟩  (one BLAS zgemm)
        moments_mat .+= Bras' * Kets
    end

    return moments_mat ./ N_randvec
end


# ------------------------------------------------------------------
# Parallel path  (Threads.@threads; each thread has its own workspace)
# ------------------------------------------------------------------
function _correlator_parallel(H_scaled, N::Int, N_randvec::Int, N_moments::Int,
                                Oa::AbstractMatrix, Ob::AbstractMatrix)
    nthreads    = Threads.maxthreadid()
    # Per-thread accumulator; reduces via sum at the end
    thread_acc  = [zeros(ComplexF64, N_moments, N_moments) for _ in 1:nthreads]

    Threads.@threads for jj in 1:N_randvec
        tid = Threads.threadid()

        r = exp.(2π * im .* rand(N))
        r ./= norm(r)

        # Thread-local Kets and Bras (allocated per-iteration)
        Kets_loc = zeros(ComplexF64, N, N_moments)
        Bras_loc = zeros(ComplexF64, N, N_moments)

        α_minus      = r
        α            = H_scaled * α_minus
        Kets_loc[:,1] .= α_minus
        Kets_loc[:,2] .= α
        for m in 3:N_moments
            α_plus        = 2 .* (H_scaled * α) .- α_minus
            Kets_loc[:,m] .= α_plus
            α_minus, α    = α, α_plus
        end
        mul!(Kets_loc, Ob, copy(Kets_loc))

        β_minus      = Oa * r
        β            = H_scaled * β_minus
        Bras_loc[:,1] .= β_minus
        Bras_loc[:,2] .= β
        for n in 3:N_moments
            β_plus        = 2 .* (H_scaled * β) .- β_minus
            Bras_loc[:,n] .= β_plus
            β_minus, β    = β, β_plus
        end

        thread_acc[tid] .+= Bras_loc' * Kets_loc
    end

    return sum(thread_acc) ./ N_randvec
end
