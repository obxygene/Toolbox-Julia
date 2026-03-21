"""
    SigmaLeads

Struct holding the self-energies for the six-terminal Hall bar.

# Fields
- `Sigma_L::Matrix{ComplexF64}`: Left longitudinal lead (W×W).
- `Sigma_R::Matrix{ComplexF64}`: Right longitudinal lead (W×W).
- `Sigma_probe::Matrix{ComplexF64}`: Transverse probe self-energy (W_probe×W_probe).
"""
struct SigmaLeads
    Sigma_L::Matrix{ComplexF64}
    Sigma_R::Matrix{ComplexF64}
    Sigma_probe::Matrix{ComplexF64}
end

"""
    six_terminal_rgf(H_layer, V_hop, W, L, W_probe, x_probe_L, x_probe_R,
                     Sigma_leads, omega; eta=1e-5)

Six-terminal Hall-bar conductance via recursive Green function.

Computes the full 6×6 transmission matrix T and Büttiker conductance matrix G
for a Hall-bar geometry with two longitudinal leads and four transverse
voltage probes.

# Lead Numbering Convention
1 = Left (source), 2 = Right (drain),
3 = Top-Left, 4 = Top-Right, 5 = Bottom-Left, 6 = Bottom-Right.

# Arguments
- `H_layer::AbstractMatrix`: (W×W) or (W×W*L) Intra-layer Hamiltonian.
- `V_hop::AbstractMatrix`: (W×W) Inter-layer hopping.
- `W::Integer`: Channel width in lattice sites.
- `L::Integer`: Number of layers.
- `W_probe::Integer`: Width of each transverse probe (≤ W/2).
- `x_probe_L::Integer`: Layer where left probes (3,5) attach.
- `x_probe_R::Integer`: Layer where right probes (4,6) attach.
- `Sigma_leads::SigmaLeads`: Self-energies of the six leads.
- `omega::Number`: Energy.
- `eta::Real=1e-5`: Infinitesimal broadening.

# Returns
- `T_mat::Matrix{Float64}`: 6×6 transmission matrix.
- `G_mat::Matrix{Float64}`: 6×6 Büttiker conductance matrix.
"""
function six_terminal_rgf(H_layer::AbstractMatrix, V_hop::AbstractMatrix,
                          W::Integer, L::Integer,
                          W_probe::Integer,
                          x_probe_L::Integer, x_probe_R::Integer,
                          Sigma_leads::SigmaLeads,
                          omega::Number;
                          eta::Real=1e-5)
    @assert W_probe <= div(W, 2) "W_probe must be ≤ floor(W/2)"
    @assert 1 <= x_probe_L <= L "x_probe_L must be in [1, L]"
    @assert 1 <= x_probe_R <= L "x_probe_R must be in [1, L]"

    Iden = Matrix{ComplexF64}(I, W, W)
    zI = (omega + im*eta) * Iden

    # Detect uniform vs non-uniform
    _, sH_c = size(H_layer)
    uniform_H = (sH_c == W)

    # Row index sets for probe coupling
    top_rows = 1:W_probe
    bot_rows = (W - W_probe + 1):W

    # Embed probe self-energies into full W×W matrices
    Sigma_p = Sigma_leads.Sigma_probe

    Sigma_TL = zeros(ComplexF64, W, W); Sigma_TL[top_rows, top_rows] .= Sigma_p
    Sigma_TR = zeros(ComplexF64, W, W); Sigma_TR[top_rows, top_rows] .= Sigma_p
    Sigma_BL = zeros(ComplexF64, W, W); Sigma_BL[bot_rows, bot_rows] .= Sigma_p
    Sigma_BR = zeros(ComplexF64, W, W); Sigma_BR[bot_rows, bot_rows] .= Sigma_p

    # Per-layer additional self-energy from transverse probes
    Sigma_extra = zeros(ComplexF64, W, W, L)
    Sigma_extra[:, :, x_probe_L] .+= Sigma_TL .+ Sigma_BL
    Sigma_extra[:, :, x_probe_R] .+= Sigma_TR .+ Sigma_BR

    # Helper: get Hamiltonian of layer x
    get_H(x) = uniform_H ? H_layer : H_layer[:, (x-1)*W+1 : x*W]

    # Recursive Dyson sweep (left to right)
    H1 = get_H(1)
    M = (zI - H1 - Sigma_extra[:,:,1] - Sigma_leads.Sigma_L) \ Iden
    G = copy(M)

    for x in 2:L
        Hx = get_H(x)
        M = (zI - Hx - Sigma_extra[:,:,x] - V_hop' * M * V_hop) \ Iden
        G = G * V_hop * M
    end

    # Right lead self-energy correction
    M = (M \ Iden - Sigma_leads.Sigma_R) \ Iden
    G_C = G + G * Sigma_leads.Sigma_R * M

    # Broadening matrices Γᵢ = i(Σᵢ - Σᵢ†)
    Gamma1 = real.(1im .* (Sigma_leads.Sigma_L .- Sigma_leads.Sigma_L'))
    Gamma2 = real.(1im .* (Sigma_leads.Sigma_R .- Sigma_leads.Sigma_R'))
    Gamma3 = real.(1im .* (Sigma_TL .- Sigma_TL'))
    Gamma4 = real.(1im .* (Sigma_TR .- Sigma_TR'))
    Gamma5 = real.(1im .* (Sigma_BL .- Sigma_BL'))
    Gamma6 = real.(1im .* (Sigma_BR .- Sigma_BR'))

    Gammas = [Gamma1, Gamma2, Gamma3, Gamma4, Gamma5, Gamma6]

    # Fisher-Lee transmission matrix
    GC_dag = G_C'
    T_mat = zeros(6, 6)

    for ii in 1:6
        GGi = Gammas[ii] * G_C
        for jj in 1:6
            ii == jj && continue
            GGj = Gammas[jj] * G_C
            T_mat[ii, jj] = real(sum(GGi .* conj.(GGj)))
        end
    end

    # Büttiker conductance matrix
    G_mat = -copy(T_mat)
    for ii in 1:6
        G_mat[ii, ii] = sum(T_mat[ii, :])
    end

    return T_mat, G_mat
end
