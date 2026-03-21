"""
    finite_temperature_transmission(omega, Transmission_0T, mu, Tem)

Finite-temperature Landauer conductance via Sommerfeld convolution.

Integrates T(E) * (-df/dE) dE where -df/dE = (1/4T) sech²((E-μ)/(2T)).

# Arguments
- `omega::AbstractVector`: Energy grid (uniform spacing assumed).
- `Transmission_0T::AbstractVector`: Zero-temperature transmission T(E).
- `mu::Real`: Chemical potential (Fermi level).
- `Tem::Real`: Temperature in energy units (k_B = 1).

# Returns
- `Trans_Tem::Float64`: Finite-temperature conductance.
"""
function finite_temperature_transmission(omega::AbstractVector,
                                         Transmission_0T::AbstractVector,
                                         mu::Real, Tem::Real)
    dE = omega[2] - omega[1]

    # Thermal broadening kernel: -df/dE = (1/4T) sech²((E-μ)/(2T))
    kernel = @. (1 / (4*Tem)) * sech((omega - mu) / (2*Tem))^2

    return sum(Transmission_0T .* kernel) * dE
end
