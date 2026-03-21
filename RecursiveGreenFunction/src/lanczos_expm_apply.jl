"""
    lanczos_expm_apply(H, v, dt; order=nothing, tol=1e-12, opts...)

Compute y = exp(i*dt*H) * v using Hermitian Lanczos + adaptive control.

# Arguments
- `H`: Hermitian matrix or function handle `x -> H*x`.
- `v::AbstractVector`: Input vector.
- `dt::Real=1.0`: Time step.
- `order::Union{Nothing,Integer}=nothing`: Fixed Krylov dimension, or `nothing` for adaptive.
- `tol::Real=1e-12`: Convergence tolerance.
- `maxDim::Integer=min(60, length(v))`: Maximum Krylov subspace dimension.
- `maxSteps::Integer=64`: Maximum time-splitting steps.
- `reorth::Symbol=:none`: Reorthogonalization: `:none` or `:full`.
- `hermitian::Bool=true`: Whether H is Hermitian.
- `verbose::Int=0`: Verbosity level.

# Returns
- `y::Vector{ComplexF64}`: Result of exp(i*dt*H)*v.
"""
function lanczos_expm_apply(H, v::AbstractVector, dt::Real=1.0;
                            order::Union{Nothing,Integer}=nothing,
                            tol::Real=1e-12,
                            maxDim::Integer=min(60, length(v)),
                            maxSteps::Integer=64,
                            reorth::Symbol=:none,
                            hermitian::Bool=true,
                            verbose::Int=0)
    # Determine operator application
    H_func = if isa(H, Function)
        H
    else
        x -> H * x
    end

    # Quick exits
    if dt == 0
        return ComplexF64.(v)
    end
    nv = norm(v)
    if nv == 0
        return ComplexF64.(v)
    end

    # Krylov dimension control
    if isnothing(order)
        m = min(30, maxDim)
        adapt_m = true
    else
        m = min(order, length(v))
        adapt_m = false
    end

    num_steps = 1

    while true
        tau = dt / num_steps
        y = ComplexF64.(v)
        ok_all = true
        max_resid_seen = 0.0

        for _ in 1:num_steps
            ys, success, resid = _krylov_exp_lanczos_step(
                H_func, y, tau, m, tol/num_steps, reorth, hermitian)
            y = ys
            max_resid_seen = max(max_resid_seen, resid)

            if !success
                ok_all = false
                break
            end
        end

        if ok_all
            if verbose > 0
                @info "lanczos: success" m num_steps max_resid_seen
            end
            return y
        end

        # Try increasing m
        if adapt_m && m < maxDim
            m_new = min(maxDim, max(ceil(Int, 1.5*m), m + 5))
            if verbose > 0
                @info "lanczos: increasing Krylov dim" m m_new
            end
            m = m_new
            continue
        end

        # Split time step
        if num_steps < maxSteps
            num_steps = min(maxSteps, max(2, 2*num_steps))
            if verbose > 0
                @info "lanczos: time-splitting" num_steps
            end
        else
            @warn "Lanczos expmv failed to meet tolerance" tol maxDim maxSteps max_resid_seen
            return y
        end
    end
end

"""
Internal: one Lanczos-Krylov step computing exp(i*tau*H)*v.
"""
function _krylov_exp_lanczos_step(Hfun, v::AbstractVector, tau::Real,
                                   m::Integer, tol::Real,
                                   reorth::Symbol, hermitian_flag::Bool)
    n = length(v)
    nv = norm(v)

    if nv == 0 || tau == 0
        return ComplexF64.(v), true, 0.0
    end

    V = zeros(ComplexF64, n, m)
    alpha = zeros(m)
    beta  = zeros(m - 1)

    V[:, 1] .= v ./ nv
    last_beta = 0.0
    j_effective = 1

    for j in 1:m
        w = Hfun(V[:, j])

        aj = if hermitian_flag
            real(dot(V[:, j], w))
        else
            dot(V[:, j], w)
        end
        alpha[j] = aj

        if j == 1
            w .-= aj .* V[:, j]
        else
            w .-= aj .* V[:, j] .+ beta[j-1] .* V[:, j-1]
        end

        # Optional reorthogonalization
        if reorth == :full
            h = V[:, 1:j]' * w
            w .-= V[:, 1:j] * h
        end

        bj = norm(w)
        last_beta = bj

        if j < m
            if bj < max(1e-14, eps()) * nv
                j_effective = j  # happy breakdown
                break
            end
            beta[j] = bj
            V[:, j+1] .= w ./ bj
            j_effective = j + 1
        else
            j_effective = j
        end
    end

    # Build tridiagonal T and compute exp(i*tau*T)*e1
    jT = j_effective
    if jT == 0
        return ComplexF64.(v), true, 0.0
    end

    T = Tridiagonal(beta[1:jT-1], alpha[1:jT], beta[1:jT-1])
    e1 = zeros(ComplexF64, jT)
    e1[1] = 1.0

    y_small = exp(Matrix(im * tau * T)) * e1

    # Residual estimate
    resid = last_beta > 0 ? abs(last_beta * y_small[end]) : 0.0
    success = resid <= tol

    # Map back to full space
    y = nv .* (V[:, 1:jT] * y_small)

    return y, success, resid
end
