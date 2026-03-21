"""
    recursive_green_function_1l(HCC, V, layer, omega, Sigma_L, Sigma_R;
                                eta=1e-5, disorder_type="Clean", Gamma=0.0, Norb=1)

Recursive retarded Green function G_{1L} with named parameters.

Cleaner refactor of `surface_green_function_gcc_1l`. All parameters are explicit.

# Physics Background
Recursive Dyson equation layer-by-layer:
  M₁ = [(ω+iη)I - H₁ - Σ_L]⁻¹
  Mᵢ = [(ω+iη)I - Hᵢ - V† Mᵢ₋₁ V]⁻¹  (i = 2,...,L)
  G_{iL} = G_{(i-1)L} * V * Mᵢ

# Arguments
- `HCC::AbstractMatrix`: (W×W) uniform or (W×W*L) concatenated Hamiltonian.
- `V::AbstractMatrix`: (W×W) uniform or (W×W*(L-1)) concatenated hopping.
- `layer::Integer`: Number of layers (uniform case).
- `omega::Number`: Energy.
- `Sigma_L::AbstractMatrix`: Left lead self-energy.
- `Sigma_R::AbstractMatrix`: Right lead self-energy.
- `eta::Real=1e-5`: Infinitesimal broadening.
- `disorder_type::String="Clean"`: "Clean", "Onsite_disorder",
  "UnitCell_disorder", or "spin_disorder".
- `Gamma::Real=0.0`: Disorder amplitude.
- `Norb::Integer=1`: Orbitals per unit cell (UnitCell_disorder only).

# Returns
- `G_1L::Matrix{ComplexF64}`: Retarded GF from layer 1 to layer L.
"""
function recursive_green_function_1l(HCC::AbstractMatrix, V::AbstractMatrix,
                                     layer::Integer, omega::Number,
                                     Sigma_L::AbstractMatrix, Sigma_R::AbstractMatrix;
                                     eta::Real=1e-5,
                                     disorder_type::String="Clean",
                                     Gamma::Real=0.0,
                                     Norb::Integer=1)
    W, L = size(HCC)
    layerflag = (W == L)
    Iden = Matrix{ComplexF64}(I, W, W)
    M_ii = zeros(ComplexF64, W, W)
    G_1L = zeros(ComplexF64, W, W)

    if layerflag
        # Uniform layers
        if disorder_type == "Onsite_disorder"
            H_imp = Diagonal((rand(W) .- 0.5) .* Gamma)
            M_ii = ((omega + im*eta) * Iden - HCC - H_imp - Sigma_L) \ Iden
            G_1L = copy(M_ii)
            for _ in 2:layer
                H_imp = Diagonal((rand(W) .- 0.5) .* Gamma)
                M_ii = ((omega + im*eta) * Iden - HCC - H_imp - V' * M_ii * V) \ Iden
                G_1L = G_1L * V * M_ii
            end

        elseif disorder_type == "UnitCell_disorder"
            n_cells = div(W, Norb)
            H_imp = Diagonal(repeat((rand(n_cells) .- 0.5) .* Gamma, inner=Norb))
            M_ii = ((omega + im*eta) * Iden - HCC - H_imp - Sigma_L) \ Iden
            G_1L = copy(M_ii)
            for _ in 2:layer
                H_imp = Diagonal(repeat((rand(n_cells) .- 0.5) .* Gamma, inner=Norb))
                M_ii = ((omega + im*eta) * Iden - HCC - H_imp - V' * M_ii * V) \ Iden
                G_1L = G_1L * V * M_ii
            end

        elseif disorder_type == "spin_disorder"
            σx = ComplexF64[0 1; 1 0]
            σy = ComplexF64[0 -im; im 0]
            H_imp = (kron(Diagonal(rand(div(W,2)) .- 0.5), σx) +
                     kron(Diagonal(rand(div(W,2)) .- 0.5), σy)) .* Gamma
            M_ii = ((omega + im*eta) * Iden - HCC - H_imp - Sigma_L) \ Iden
            G_1L = copy(M_ii)
            for _ in 2:layer
                H_imp = (kron(Diagonal(rand(div(W,2)) .- 0.5), σx) +
                         kron(Diagonal(rand(div(W,2)) .- 0.5), σy)) .* Gamma
                M_ii = ((omega + im*eta) * Iden - HCC - H_imp - V' * M_ii * V) \ Iden
                G_1L = G_1L * V * M_ii
            end

        else  # "Clean"
            M_ii = ((omega + im*eta) * Iden - HCC - Sigma_L) \ Iden
            G_1L = copy(M_ii)
            for _ in 2:layer
                M_ii = ((omega + im*eta) * Iden - HCC - V' * M_ii * V) \ Iden
                G_1L = G_1L * V * M_ii
            end
        end

        # Right self-energy correction
        M_ii = (M_ii \ Iden - Sigma_R) \ Iden
        G_1L = G_1L + G_1L * Sigma_R * M_ii

    else
        # Non-uniform layers
        nL = div(L, W)

        M_ii = ((omega + im*eta) * Iden - HCC[:, 1:W] - Sigma_L) \ Iden
        G_1L = copy(M_ii)

        for ii in 2:nL
            H_ii = HCC[:, (ii-1)*W+1 : ii*W]
            V_ii = V[:,   (ii-2)*W+1 : (ii-1)*W]
            M_ii = ((omega + im*eta) * Iden - H_ii - V_ii' * M_ii * V_ii) \ Iden
            G_1L = G_1L * V_ii * M_ii
        end

        M_ii = (M_ii \ Iden - Sigma_R) \ Iden
        G_1L = G_1L + G_1L * Sigma_R * M_ii
    end

    return G_1L
end
