"""
    tridiag_block_inv(a, b, m, n)
    tridiag_block_inv(a_single, b_single, m, n, N_block)

Return the (m,n) block of the inverse of a block-tridiagonal matrix.

Uses a backward LDU-type recursion: D_i = a_i - U_i * b_i', cost O(N_block).

# Arguments (general case)
- `a::Array{<:Number, 3}`: (blocksize × blocksize × N_block) diagonal blocks.
- `b::Array{<:Number, 3}`: (blocksize × blocksize × N_block-1) off-diagonal blocks.
- `m::Integer`: Target block row (1-indexed).
- `n::Integer`: Target block column (1-indexed).

# Arguments (uniform case)
- `a_single::AbstractMatrix`: Single diagonal block repeated N_block times.
- `b_single::AbstractMatrix`: Single off-diagonal block repeated.
- `m::Integer`: Target block row.
- `n::Integer`: Target block column.
- `N_block::Integer`: Number of blocks.

# Returns
- `inv_target_block::Matrix{ComplexF64}`: The requested inverse block.
"""
function tridiag_block_inv(a::Array{T1,3}, b::Array{T2,3},
                           m::Integer, n::Integer) where {T1<:Number, T2<:Number}
    blocksize, _, N_block = size(a)

    D = zeros(ComplexF64, blocksize, blocksize, N_block)
    U = zeros(ComplexF64, blocksize, blocksize, N_block - 1)
    L_mat = zeros(ComplexF64, blocksize, blocksize, N_block - 1)

    # Backward recursion
    D[:, :, N_block] .= a[:, :, N_block]
    for ii in (N_block-1):-1:1
        U[:, :, ii] .= b[:, :, ii] / D[:, :, ii+1]
        L_mat[:, :, ii] .= D[:, :, ii+1] \ b[:, :, ii]'
        D[:, :, ii] .= a[:, :, ii] .- U[:, :, ii] * b[:, :, ii]'
    end

    # Invert all diagonal blocks
    Dinv = similar(D)
    for ii in 1:N_block
        Dinv[:, :, ii] .= inv(D[:, :, ii])
    end

    Ainv = copy(Dinv[:, :, 1])

    if m == n
        for ii in 2:m
            Ainv = Dinv[:, :, ii] + L_mat[:, :, ii-1] * Ainv * U[:, :, ii-1]
        end
        return Ainv
    elseif m > n
        for ii in 2:n
            Ainv = Dinv[:, :, ii] + L_mat[:, :, ii-1] * Ainv * U[:, :, ii-1]
        end
        for ii in (n+1):m
            Ainv = -L_mat[:, :, ii-1] * Ainv
        end
        return Ainv
    else  # m < n
        for ii in 2:m
            Ainv = Dinv[:, :, ii] + L_mat[:, :, ii-1] * Ainv * U[:, :, ii-1]
        end
        for ii in (m+1):n
            Ainv = -Ainv * U[:, :, ii-1]
        end
        return Ainv
    end
end

# Uniform chain variant
function tridiag_block_inv(a::AbstractMatrix, b::AbstractMatrix,
                           m::Integer, n::Integer, N_block::Integer)
    blocksize = size(a, 1)

    D = zeros(ComplexF64, blocksize, blocksize, N_block)
    U = zeros(ComplexF64, blocksize, blocksize, N_block - 1)
    L_mat = zeros(ComplexF64, blocksize, blocksize, N_block - 1)

    D[:, :, N_block] .= a
    for ii in (N_block-1):-1:1
        U[:, :, ii] .= b / D[:, :, ii+1]
        L_mat[:, :, ii] .= D[:, :, ii+1] \ b'
        D[:, :, ii] .= a .- U[:, :, ii] * b'
    end

    Dinv = similar(D)
    for ii in 1:N_block
        Dinv[:, :, ii] .= inv(D[:, :, ii])
    end

    Ainv = copy(Dinv[:, :, 1])

    if m == n
        for ii in 2:m
            Ainv = Dinv[:, :, ii] + L_mat[:, :, ii-1] * Ainv * U[:, :, ii-1]
        end
        return Ainv
    elseif m > n
        for ii in 2:n
            Ainv = Dinv[:, :, ii] + L_mat[:, :, ii-1] * Ainv * U[:, :, ii-1]
        end
        for ii in (n+1):m
            Ainv = -L_mat[:, :, ii-1] * Ainv
        end
        return Ainv
    else
        for ii in 2:m
            Ainv = Dinv[:, :, ii] + L_mat[:, :, ii-1] * Ainv * U[:, :, ii-1]
        end
        for ii in (m+1):n
            Ainv = -Ainv * U[:, :, ii-1]
        end
        return Ainv
    end
end
