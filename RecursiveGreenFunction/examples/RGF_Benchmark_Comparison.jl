"""
RGF_Benchmark_Comparison.jl

Comprehensive benchmark of the Recursive Green Function toolkit in Julia.
Runs the same 5 tasks as the companion MATLAB script RGF_Benchmark.m, then
loads the MATLAB timing/accuracy CSV files (if present) and produces a
head-to-head comparison table and plots.

Run
---
  julia examples/RGF_Benchmark_Comparison.jl   (from RecursiveGreenFunction/)

Workflow
--------
  1. Run RGF_Benchmark.m in MATLAB first -- writes CSV files to benchmark_results/
  2. Run this script -- reads those CSVs, adds Julia results, prints + plots

Tasks
-----
  1. Surface GF single call (Lopez-Sancho V2)
  2. Full RGF pipeline, single energy point
  3. Full transmission spectrum (N_point energy points)
  4. Layer-count scaling (vary L, fixed W)
  5. Width scaling (vary W, fixed L)
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using RecursiveGreenFunction
using LinearAlgebra, Statistics, Printf, Plots

# ── Shared output directory ───────────────────────────────────────────────────
const BENCH_DIR = joinpath(@__DIR__, "..", "benchmark_results")
mkpath(BENCH_DIR)

# ── CSV helpers (defined at top level so they are available everywhere) ───────
function read_timing_csv(path)
    d = Dict{String,Float64}()
    for line in eachline(path)
        startswith(line, "task") && continue
        parts = split(line, ",")
        length(parts) >= 2 || continue
        d[parts[1]] = parse(Float64, parts[2])
    end
    return d
end

function read_spectrum_csv(path)
    omega_v = Float64[];  T_v = Float64[]
    for line in eachline(path)
        startswith(line, "omega") && continue
        parts = split(line, ",")
        length(parts) == 2 || continue
        push!(omega_v, parse(Float64, parts[1]))
        push!(T_v,     parse(Float64, parts[2]))
    end
    return omega_v, T_v
end

function read_scaling_csv(path)
    xs = Int[];  ys = Float64[]
    for line in eachline(path)
        (startswith(line, "layer") || startswith(line, "width")) && continue
        parts = split(line, ",")
        length(parts) == 2 || continue
        push!(xs, parse(Int,     parts[1]))
        push!(ys, parse(Float64, parts[2]))
    end
    return xs, ys
end

# ── Single-channel analytical formula (for Julia-only accuracy check) ─────────
function T_1ch_analytical(omega::Real, t::Real, tr::Real)
    abs(omega) >= 2t && return 0.0
    num  = t^4 * tr^4 * (4t^2 - omega^2)
    den1 = t^2 * omega^2 + tr^4 - tr^2 * omega^2
    den2 = 4t^6 + omega^2*(-4t^4 + 2t^2*tr^2 + tr^4) + omega^4*(t^2 - tr^2)
    d = den1 * den2
    d == 0.0 && return 0.0
    return num / d
end

# =============================================================================
# main() — all benchmark logic lives here so variables are local
# =============================================================================
function main()

    # ── Parameters (must match RGF_Benchmark.m exactly) ──────────────────────
    Width    = 50
    n_layer  = 201
    N_point  = 401
    t_hop    = 1.0
    Ec       = 0.0
    eta      = 1e-5
    Accuracy = 1e-15
    Tr       = 1.0       # perfect contacts

    energyscale  = 4.0 * t_hop
    omega_range  = range(-energyscale, energyscale, N_point)
    layer_sizes  = [10, 50, 100, 200, 500, 1000]
    width_sizes  = [5, 10, 20, 50, 100]

    # ── Hamiltonians ──────────────────────────────────────────────────────────
    H00 = Matrix{ComplexF64}(
        Ec .* I(Width) .-
        t_hop .* (diagm(1 => ones(Width-1)) .+ diagm(-1 => ones(Width-1))))
    V   = -t_hop .* Matrix{ComplexF64}(I(Width))
    HCC = copy(H00)

    println("=" ^ 70)
    println("  RGF Benchmark - Julia")
    println("  Width=$Width  Layers=$n_layer  N_point=$N_point  t=$t_hop  Tr=$Tr")
    println("=" ^ 70)

    # ── Warm-up (trigger JIT before timing) ───────────────────────────────────
    println("\n[Warm-up] Compiling Julia functions...")
    let w = 0.0
        G00R, _ = surface_green_function_v2(H00, V,  w, Accuracy; eta=eta)
        G00L, _ = surface_green_function_v2(H00, V', w, Accuracy; eta=eta)
        SL = surface_green_function_self_energy(G00L, Tr .* V')
        SR = surface_green_function_self_energy(G00R, Tr .* V)
        GL = surface_green_function_broadening(SL)
        GR = surface_green_function_broadening(SR)
        G1L = recursive_green_function_1l(HCC, V, n_layer, w, SL, SR; eta=eta)
        real(green_function_transmission(GL, GR, G1L))
    end
    println("  Done.\n")

    # =========================================================================
    # BENCHMARK 1 - Single surface GF call
    # =========================================================================
    println("-" ^ 70)
    println("BENCHMARK 1: Single surface GF call (Lopez-Sancho V2)")
    println("-" ^ 70)

    N_rep1 = 20
    t1_total = @elapsed for _ in 1:N_rep1
        surface_green_function_v2(H00, V, 0.0, Accuracy; eta=eta)
    end
    t1_ms = t1_total / N_rep1 * 1e3
    @printf("  Julia: %7.3f ms/call  (avg over %d calls)\n\n", t1_ms, N_rep1)

    # =========================================================================
    # BENCHMARK 2 - Full RGF pipeline, single energy point
    # =========================================================================
    println("-" ^ 70)
    println("BENCHMARK 2: Full RGF pipeline (surface GF + G_{1L} + T)")
    println("-" ^ 70)

    N_rep2 = 5
    t2_total = @elapsed for _ in 1:N_rep2
        G00R, _ = surface_green_function_v2(H00, V,  0.0, Accuracy; eta=eta)
        G00L, _ = surface_green_function_v2(H00, V', 0.0, Accuracy; eta=eta)
        SL = surface_green_function_self_energy(G00L, Tr .* V')
        SR = surface_green_function_self_energy(G00R, Tr .* V)
        GL = surface_green_function_broadening(SL)
        GR = surface_green_function_broadening(SR)
        G1L = recursive_green_function_1l(HCC, V, n_layer, 0.0, SL, SR; eta=eta)
        real(green_function_transmission(GL, GR, G1L))
    end
    t2_ms = t2_total / N_rep2 * 1e3
    @printf("  Julia: %7.3f ms/call  (avg over %d calls)\n\n", t2_ms, N_rep2)

    # =========================================================================
    # BENCHMARK 3 - Full transmission spectrum
    # =========================================================================
    println("-" ^ 70)
    println("BENCHMARK 3: Full transmission spectrum ($N_point energy points)")
    println("-" ^ 70)

    T_julia = zeros(Float64, N_point)
    t3_s = @elapsed for (ii, w) in enumerate(omega_range)
        G00R, _ = surface_green_function_v2(H00, V,  w, Accuracy; eta=eta)
        G00L, _ = surface_green_function_v2(H00, V', w, Accuracy; eta=eta)
        SL = surface_green_function_self_energy(G00L, Tr .* V')
        SR = surface_green_function_self_energy(G00R, Tr .* V)
        GL = surface_green_function_broadening(SL)
        GR = surface_green_function_broadening(SR)
        G1L = recursive_green_function_1l(HCC, V, n_layer, w, SL, SR; eta=eta)
        T_julia[ii] = real(green_function_transmission(GL, GR, G1L))
    end
    @printf("  Julia: %7.3f s total  (%6.2f ms/point)\n\n", t3_s, t3_s/N_point*1e3)

    # =========================================================================
    # BENCHMARK 4 - Layer scaling
    # =========================================================================
    println("-" ^ 70)
    println("BENCHMARK 4: Layer scaling  (W=$Width, omega=0)")
    println("-" ^ 70)

    t4_ms = zeros(length(layer_sizes))

    G00R0, _ = surface_green_function_v2(H00, V,  0.0, Accuracy; eta=eta)
    G00L0, _ = surface_green_function_v2(H00, V', 0.0, Accuracy; eta=eta)
    SL0 = surface_green_function_self_energy(G00L0, Tr .* V')
    SR0 = surface_green_function_self_energy(G00R0, Tr .* V)

    @printf("  %8s  %12s\n", "Layers", "Time (ms)")
    println("  " * "-"^22)
    for (jj, L_test) in enumerate(layer_sizes)
        N_rep = max(5, min(200, round(Int, 1000 / L_test)))
        tt = @elapsed for _ in 1:N_rep
            recursive_green_function_1l(HCC, V, L_test, 0.0, SL0, SR0; eta=eta)
        end
        t4_ms[jj] = tt / N_rep * 1e3
        @printf("  %8d  %12.3f\n", L_test, t4_ms[jj])
    end
    println()

    # =========================================================================
    # BENCHMARK 5 - Width scaling
    # =========================================================================
    println("-" ^ 70)
    println("BENCHMARK 5: Width scaling  (L=$n_layer, omega=0)")
    println("-" ^ 70)

    t5_ms = zeros(length(width_sizes))

    @printf("  %8s  %12s\n", "Width", "Time (ms)")
    println("  " * "-"^22)
    for (jj, W_test) in enumerate(width_sizes)
        H00_w = Matrix{ComplexF64}(
            Ec .* I(W_test) .-
            t_hop .* (diagm(1 => ones(W_test-1)) .+ diagm(-1 => ones(W_test-1))))
        V_w = -t_hop .* Matrix{ComplexF64}(I(W_test))

        G00R_w, _ = surface_green_function_v2(H00_w, V_w,  0.0, Accuracy; eta=eta)
        G00L_w, _ = surface_green_function_v2(H00_w, V_w', 0.0, Accuracy; eta=eta)
        SL_w = surface_green_function_self_energy(G00L_w, Tr .* V_w')
        SR_w = surface_green_function_self_energy(G00R_w, Tr .* V_w)

        # warm-up for this width
        recursive_green_function_1l(H00_w, V_w, n_layer, 0.0, SL_w, SR_w; eta=eta)

        N_rep = max(5, min(100, round(Int, 500*100 / W_test^2)))
        tt = @elapsed for _ in 1:N_rep
            recursive_green_function_1l(H00_w, V_w, n_layer, 0.0, SL_w, SR_w; eta=eta)
        end
        t5_ms[jj] = tt / N_rep * 1e3
        @printf("  %8d  %12.3f\n", W_test, t5_ms[jj])
    end
    println()

    # =========================================================================
    # EXPORT Julia timings to CSV
    # =========================================================================
    open(joinpath(BENCH_DIR, "benchmark_timings_julia.csv"), "w") do f
        println(f, "task,time_ms,n_repeats,notes")
        println(f, "surface_gf_single,$t1_ms,$N_rep1,\"W=$Width Lopez-Sancho V2\"")
        println(f, "rgf_pipeline_single,$t2_ms,$N_rep2,\"W=$Width L=$n_layer full pipeline\"")
        println(f, "spectrum_total_s,$t3_s,1,\"$N_point pts W=$Width L=$n_layer\"")
        println(f, "spectrum_per_point_ms,$(t3_s/N_point*1e3),$N_point,\"\"")
    end

    open(joinpath(BENCH_DIR, "transmission_spectrum_julia.csv"), "w") do f
        println(f, "omega,transmission")
        for (w, T) in zip(omega_range, T_julia)
            @printf(f, "%.10f,%.10f\n", w, T)
        end
    end

    open(joinpath(BENCH_DIR, "layer_scaling_julia.csv"), "w") do f
        println(f, "layers,time_ms")
        for (L, tm) in zip(layer_sizes, t4_ms)
            @printf(f, "%d,%.6f\n", L, tm)
        end
    end

    open(joinpath(BENCH_DIR, "width_scaling_julia.csv"), "w") do f
        println(f, "width,time_ms")
        for (W, tm) in zip(width_sizes, t5_ms)
            @printf(f, "%d,%.6f\n", W, tm)
        end
    end

    println("Julia results exported to benchmark_results/")

    # =========================================================================
    # COMPARISON: load MATLAB results if available
    # =========================================================================
    matlab_available = isfile(joinpath(BENCH_DIR, "benchmark_timings_matlab.csv"))

    if matlab_available
        m_timing  = read_timing_csv(joinpath(BENCH_DIR, "benchmark_timings_matlab.csv"))
        omega_m, T_m = read_spectrum_csv(joinpath(BENCH_DIR, "transmission_spectrum_matlab.csv"))
        L_m,  tL_m   = read_scaling_csv(joinpath(BENCH_DIR, "layer_scaling_matlab.csv"))
        W_m,  tW_m   = read_scaling_csv(joinpath(BENCH_DIR, "width_scaling_matlab.csv"))

        m1 = get(m_timing, "surface_gf_single",   NaN)
        m2 = get(m_timing, "rgf_pipeline_single", NaN)
        m3 = get(m_timing, "spectrum_total_s",    NaN)

        # Interpolate MATLAB spectrum onto Julia omega grid
        T_m_interp = length(T_m) == N_point ? T_m :
            [T_m[clamp(searchsortedfirst(omega_m, w), 1, length(T_m))]
             for w in omega_range]

        max_err  = maximum(abs.(T_julia .- T_m_interp))
        mean_err = mean(abs.(T_julia .- T_m_interp))

        # Summary table
        println("\n" * "=" ^ 70)
        println("  COMPARISON: Julia vs MATLAB  (W=$Width, L=$n_layer)")
        println("=" ^ 70)
        @printf("  %-38s  %9s  %9s  %7s\n", "Task", "Julia", "MATLAB", "Speedup")
        println("  " * "-"^68)
        @printf("  %-38s  %7.3f ms  %7.3f ms  %6.1fx\n",
            "Surface GF single (W=$Width)", t1_ms, m1, m1/t1_ms)
        @printf("  %-38s  %7.3f ms  %7.3f ms  %6.1fx\n",
            "Full RGF pipeline (W=$Width,L=$n_layer)", t2_ms, m2, m2/t2_ms)
        @printf("  %-38s  %7.3f s   %7.3f s   %6.1fx\n",
            "Full spectrum ($N_point pts)", t3_s, m3, m3/t3_s)
        println("  " * "-"^68)
        @printf("  %-38s  max=%.2e  mean=%.2e\n",
            "Accuracy |T_Julia - T_MATLAB|", max_err, mean_err)
        println("=" ^ 70)

        # Plots
        omega_v = collect(omega_range)
        speedups = [m1/t1_ms, m2/t2_ms, m3/t3_s]

        p_spec = plot(omega_v, T_julia;
            label="Julia", lw=1.8, color=:steelblue,
            xlabel="Energy", ylabel="T(omega)",
            title="Transmission Spectrum\n(W=$Width, L=$n_layer)",
            ylims=(-0.3, Width+1), xlims=(-energyscale, energyscale),
            grid=true, framestyle=:box)
        plot!(p_spec, omega_m, T_m;
            label="MATLAB", lw=1.8, color=:tomato, linestyle=:dash)
        hline!(p_spec, [Width]; ls=:dot, color=:gray40, lw=1, label="T=W=$Width")

        p_err = plot(omega_v, abs.(T_julia .- T_m_interp);
            label="|T_Julia - T_MATLAB|",
            xlabel="Energy", ylabel="|delta T|",
            title="Pointwise Error\n(max=$(round(max_err, sigdigits=2)))",
            lw=1.5, color=:purple,
            yscale=:log10, ylims=(1e-16, 1e-1),
            grid=true, framestyle=:box)

        p_layer = plot(L_m, tL_m;
            label="MATLAB", lw=1.8, color=:tomato, marker=:square, markersize=5,
            xlabel="Layers L", ylabel="Time (ms)",
            title="Layer Scaling (W=$Width)",
            xscale=:log10, yscale=:log10, grid=true, framestyle=:box)
        plot!(p_layer, layer_sizes, t4_ms;
            label="Julia", lw=1.8, color=:steelblue, marker=:circle, markersize=5)

        p_width = plot(W_m, tW_m;
            label="MATLAB", lw=1.8, color=:tomato, marker=:square, markersize=5,
            xlabel="Width W", ylabel="Time (ms)",
            title="Width Scaling (L=$n_layer)",
            xscale=:log10, yscale=:log10, grid=true, framestyle=:box)
        plot!(p_width, width_sizes, t5_ms;
            label="Julia", lw=1.8, color=:steelblue, marker=:circle, markersize=5)

        p_bar = bar(["Surface GF", "RGF Pipeline", "Spectrum"], speedups;
            ylabel="Speedup (MATLAB / Julia)",
            title="Julia Speedup over MATLAB",
            color=[:steelblue, :mediumseagreen, :darkorange],
            legend=false, grid=true, framestyle=:box,
            ylims=(0, max(maximum(speedups)*1.35, 2.0)))
        hline!(p_bar, [1.0]; ls=:dash, color=:gray40, lw=1.2, label=false)
        for (i, s) in enumerate(speedups)
            annotate!(p_bar, i, s + 0.06*maximum(speedups),
                text(@sprintf("%.1fx", s), 9, :center))
        end

        fig = plot(p_spec, p_err, p_layer, p_width, p_bar;
            layout=@layout([a b c; d e _]),
            size=(1400, 700),
            plot_title="Julia vs MATLAB -- RGF Benchmark (W=$Width, L=$n_layer)",
            left_margin=5Plots.mm, bottom_margin=7Plots.mm, top_margin=3Plots.mm)

        savefig(fig, joinpath(BENCH_DIR, "benchmark_comparison.png"))
        println("\nComparison plot -> benchmark_results/benchmark_comparison.png")
        display(fig)

    else
        # Julia-only: compare against analytical formula
        println("\n" * "=" ^ 70)
        println("  Julia-only results  (MATLAB CSV not found in benchmark_results/)")
        println("  Run RGF_Benchmark.m in MATLAB first for a head-to-head comparison.")
        println("=" ^ 70)
        @printf("  %-40s  %8.3f ms\n", "Surface GF single (W=$Width)",           t1_ms)
        @printf("  %-40s  %8.3f ms\n", "Full RGF pipeline (W=$Width,L=$n_layer)", t2_ms)
        @printf("  %-40s  %8.3f s\n",  "Full spectrum ($N_point pts)",             t3_s)
        println("=" ^ 70)

        EigenEnergy = [-2t_hop * cos(n*pi/(Width+1)) for n in 1:Width]
        T_ana = [sum(T_1ch_analytical(w - En, t_hop, Tr*t_hop) for En in EigenEnergy)
                 for w in omega_range]
        max_err_ana  = maximum(abs.(T_julia .- T_ana))
        mean_err_ana = mean(abs.(T_julia .- T_ana))
        @printf("\n  Accuracy vs analytical:  max=%.2e  mean=%.2e\n",
            max_err_ana, mean_err_ana)

        omega_v = collect(omega_range)

        p_spec = plot(omega_v, T_julia;
            label="Julia (RGF)", lw=1.8, color=:steelblue,
            xlabel="Energy", ylabel="T(omega)",
            title="Transmission  W=$Width, L=$n_layer, t=$t_hop, Tr=$Tr",
            ylims=(-0.3, Width+1), grid=true, framestyle=:box)
        plot!(p_spec, omega_v, T_ana;
            label="Analytical", lw=1.5, color=:tomato, linestyle=:dash)
        hline!(p_spec, [Width]; ls=:dot, color=:gray40, lw=1, label="T=W")

        p_layer = plot(layer_sizes, t4_ms;
            label="Julia", lw=1.8, color=:steelblue, marker=:circle,
            xlabel="Layers L", ylabel="Time (ms)",
            title="Layer Scaling (W=$Width)",
            xscale=:log10, yscale=:log10, grid=true, framestyle=:box)

        p_width = plot(width_sizes, t5_ms;
            label="Julia", lw=1.8, color=:steelblue, marker=:circle,
            xlabel="Width W", ylabel="Time (ms)",
            title="Width Scaling (L=$n_layer)",
            xscale=:log10, yscale=:log10, grid=true, framestyle=:box)

        fig = plot(p_spec, p_layer, p_width;
            layout=(1,3), size=(1200, 400),
            plot_title="Julia RGF Benchmark  W=$Width, L=$n_layer",
            left_margin=5Plots.mm, bottom_margin=7Plots.mm)

        savefig(fig, joinpath(BENCH_DIR, "benchmark_julia_only.png"))
        println("Plot -> benchmark_results/benchmark_julia_only.png")
        display(fig)
    end
end

main()
