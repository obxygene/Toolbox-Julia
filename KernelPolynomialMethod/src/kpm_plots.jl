# ============================================================
# kpm_plots.jl  –  Convenience plotting helpers using Plots.jl
# ============================================================

"""
    plot_dos(omega, dos; title="Density of States", kwargs...)

Plot the DOS curve.  Any extra `kwargs` are forwarded to `plot!`.
"""
function plot_dos(omega, dos; title::String = "Density of States", kw...)
    p = plot(omega, dos;
             xlabel = "Energy",
             ylabel = "DOS",
             title  = title,
             lw     = 1.5,
             legend = false,
             kw...)
    return p
end


"""
    plot_dos_compare(omega_kpm, dos_kpm, omega_exact, dos_exact;
                     title="DOS: KPM vs Exact", kwargs...)

Overlay KPM and exact (or analytic) DOS curves.
"""
function plot_dos_compare(omega_kpm, dos_kpm, omega_exact, dos_exact;
                          title::String = "DOS: KPM vs Exact", kw...)
    p = plot(omega_kpm,  dos_kpm  ./ maximum(dos_kpm);
             label  = "KPM",
             lw     = 1.5,
             lc     = :blue,
             xlabel = "Energy",
             ylabel = "DOS (normalised)",
             title  = title)
    plot!(p, omega_exact, dos_exact ./ maximum(dos_exact);
          label = "Analytic",
          lw    = 1.5,
          lc    = :red,
          ls    = :dash,
          kw...)
    return p
end


"""
    plot_conductance(E_fermi, sigma_xy, sigma_xx;
                     plateaux=Int[], title="Hall & Longitudinal Conductance")

Three-panel figure: σ_xy (left) and σ_xx (right), with optional horizontal
dashed lines at the expected quantized plateaux values.
"""
function plot_conductance(E_fermi, sigma_xy, sigma_xx;
                          plateaux::AbstractVector = Int[],
                          title::String = "Hall & Longitudinal Conductance",
                          label::String = "KPM")
    p_xy = plot(E_fermi, sigma_xy;
                xlabel = "Fermi energy  E_F",
                ylabel = "σ_xy  (e²/h)",
                title  = "Hall conductivity σ_xy",
                lw     = 1.5,
                lc     = :red,
                label  = label)
    if !isempty(plateaux)
        hline!(p_xy, Float64.(plateaux);
               ls = :dash, lc = :black, alpha = 0.3, label = "")
    end

    p_xx = plot(E_fermi, sigma_xx;
                xlabel = "Fermi energy  E_F",
                ylabel = "σ_xx  (e²/h)",
                title  = "Longitudinal conductivity σ_xx",
                lw     = 1.5,
                lc     = :blue,
                label  = label)

    return plot(p_xy, p_xx; layout = (1, 2), plot_title = title, size = (1100, 430))
end


"""
    plot_conductance_compare(E, sigma_xy_kpm, sigma_xy_exact,
                              sigma_xx_kpm, sigma_xx_exact;
                              plateaux=Int[], title="KPM vs Exact")

Overlay KPM and exact conductivities side by side.
"""
function plot_conductance_compare(E, sigma_xy_kpm, sigma_xy_exact,
                                   sigma_xx_kpm, sigma_xx_exact;
                                   plateaux::AbstractVector = Int[],
                                   title::String = "KPM vs Exact")
    p_xy = plot(E, sigma_xy_exact; label = "Exact", lw = 2,  lc = :blue,
                xlabel = "E_F", ylabel = "σ_xy  (e²/h)", title = "σ_xy")
    plot!(p_xy, E, sigma_xy_kpm;  label = "KPM",   lw = 1.5, lc = :red, ls = :dash)
    if !isempty(plateaux)
        hline!(p_xy, Float64.(plateaux); ls = :dot, lc = :black, alpha = 0.3, label = "")
    end

    p_xx = plot(E, sigma_xx_exact; label = "Exact", lw = 2,  lc = :blue,
                xlabel = "E_F", ylabel = "σ_xx  (e²/h)", title = "σ_xx")
    plot!(p_xx, E, sigma_xx_kpm;  label = "KPM",   lw = 1.5, lc = :red, ls = :dash)

    # Ratio panel
    ratio = sigma_xy_kpm ./ sigma_xy_exact
    valid = isfinite.(ratio) .& (abs.(ratio) .< 100) .& (abs.(sigma_xy_exact) .> 0.5)
    p_r = scatter(E[valid], ratio[valid];
                  ms = 2, label = "", xlabel = "E_F",
                  ylabel = "KPM/Exact", title = "σ_xy ratio")
    hline!(p_r, [1.0]; lw = 1.5, lc = :blue, label = "ideal = 1")

    return plot(p_xy, p_xx, p_r; layout = (1, 3),
                plot_title = title, size = (1400, 430))
end


"""
    plot_convergence(E_fermi, results;
                     quantity=:sigma_xy,
                     plateaux=Int[],
                     title="Convergence with M")

Overlay multiple σ curves (one per entry in `results`) to show M-convergence.
`results` is a vector of NamedTuples with fields `M`, `SR`, `sigma_xy`, `sigma_xx`.
`quantity` is `:sigma_xy` or `:sigma_xx`.
"""
function plot_convergence(E_fermi, results;
                          quantity::Symbol = :sigma_xy,
                          plateaux::AbstractVector = Int[],
                          title::String = "Convergence with M")
    colors = [:blue, :red, :black, :green, :orange]
    ylab   = (quantity == :sigma_xy) ? "σ_xy  (e²/h)" : "σ_xx  (e²/h)"
    p      = plot(; xlabel = "Fermi energy  E_F", ylabel = ylab,
                    title = title, legend = :topright)
    for (i, r) in enumerate(results)
        y = (quantity == :sigma_xy) ? r.sigma_xy : r.sigma_xx
        c = colors[mod1(i, length(colors))]
        plot!(p, E_fermi, y; lw = 1.5, lc = c,
              label = "M=$(r.M), R=$(r.SR)")
    end
    if !isempty(plateaux)
        hline!(p, Float64.(plateaux); ls = :dash, lc = :black, alpha = 0.25, label = "")
    end
    return p
end
