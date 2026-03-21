"""
    surface_green_function_gcc_1l_add_end_layer(HCC, V, layer, omega,
        Sigma_L, Sigma_R, HCC_end, VCC_end; eta=1e-5)

Recursive G_{1L} with a distinct final (right boundary) layer.

Extension of `surface_green_function_gcc_1l` for devices where the last
layer has a different Hamiltonian and/or coupling than the bulk layers.

# Arguments
- `HCC::AbstractMatrix`: (W×W) or (W×W*(L-1)) bulk layer Hamiltonian.
- `V::AbstractMatrix`: (W×W) or (W×W*(L-2)) bulk hopping.
- `layer::Integer`: Total number of layers including end layer.
- `omega::Number`: Energy.
- `Sigma_L::AbstractMatrix`: Left lead self-energy.
- `Sigma_R::AbstractMatrix`: Right lead self-energy.
- `HCC_end::AbstractMatrix`: (W×W) Hamiltonian of the right boundary layer.
- `VCC_end::AbstractMatrix`: (W×W) Hopping coupling to the end layer.
- `eta::Real=1e-5`: Infinitesimal broadening.

# Returns
- `Gcc_1L::Matrix{ComplexF64}`: G_{1L} connecting leftmost to rightmost layer.
"""
function surface_green_function_gcc_1l_add_end_layer(
        HCC::AbstractMatrix, V::AbstractMatrix,
        layer::Integer, omega::Number,
        Sigma_L::AbstractMatrix, Sigma_R::AbstractMatrix,
        HCC_end::AbstractMatrix, VCC_end::AbstractMatrix;
        eta::Real=1e-5)

    W, L = size(HCC)
    layerflag = (W == L)
    Iden = Matrix{ComplexF64}(I, W, W)

    if layerflag
        # Uniform bulk layers
        M_ii = ((omega + im*eta) * Iden - HCC - Sigma_L) \ Iden
        Gcc_1L = copy(M_ii)

        for _ in 2:(layer-1)
            M_ii = ((omega + im*eta) * Iden - HCC - V' * M_ii * V) \ Iden
            Gcc_1L = Gcc_1L * V * M_ii
        end

        # Final (end) layer with HCC_end and VCC_end, includes Sigma_R
        M_ii = ((omega + im*eta) * Iden - HCC_end - VCC_end' * M_ii * VCC_end - Sigma_R) \ Iden
        Gcc_1L = Gcc_1L * VCC_end * M_ii
    else
        # Non-uniform layers
        nL = div(L, W)

        M_ii = ((omega + im*eta) * Iden - HCC[:, 1:W] - Sigma_L) \ Iden
        Gcc_1L = copy(M_ii)

        for ii in 2:(nL - 1)
            H_ii = HCC[:, (ii-1)*W+1 : ii*W]
            V_ii = V[:,   (ii-2)*W+1 : (ii-1)*W]
            M_ii = ((omega + im*eta) * Iden - H_ii - V_ii' * M_ii * V_ii) \ Iden
            Gcc_1L = Gcc_1L * V_ii * M_ii
        end

        # Final layer from the last block of HCC
        H_ii = HCC[:, end-W+1:end]
        V_ii = V[:, (nL-2)*W+1 : (nL-1)*W]
        M_ii = ((omega + im*eta) * Iden - H_ii - V_ii' * M_ii * V_ii - Sigma_R) \ Iden
        Gcc_1L = Gcc_1L * V_ii * M_ii
    end

    return Gcc_1L
end
