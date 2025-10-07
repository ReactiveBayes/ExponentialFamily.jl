using BenchmarkTools
using LinearAlgebra
using Distributions
using BayesBase
using ExponentialFamily
using FastCholesky
using StatsFuns: log2π


# Implementation 1: Using cholinv_logdet and dot3arg
function compute_logscale_v1(
    left::L, right::R
) where {
    L <: MultivariateNormalDistributionsFamily,
    R <: MultivariateNormalDistributionsFamily
}
    m_left, v_left   = mean_cov(left)
    m_right, v_right = mean_cov(right)
    v                = v_left + v_right
    n                = length(left)
    v_inv, v_logdet  = cholinv_logdet(v)
    m                = m_left - m_right
    return -(v_logdet + n * log2π) / 2 - dot(m, v_inv, m) / 2
end

# Implementation 2: Using logdet and backslash operator
function compute_logscale_v2(
    left::L, right::R
) where {
    L <: MultivariateNormalDistributionsFamily,
    R <: MultivariateNormalDistributionsFamily
}
    m_left, v_left   = mean_cov(left)
    m_right, v_right = mean_cov(right)
    v                = v_left + v_right
    n                = length(left)
    m                = m_left - m_right
    return -(logdet(v) + n * log2π) / 2 - dot(m, v \ m) / 2
end

# Test function
function benchmark_compute_logscale_normalxnormal()
    println("=" ^ 70)
    println("Benchmarking compute_logscale implementations")
    println("=" ^ 70)
    
    # Test different dimensionalities
    dims = [2, 5, 10, 20, 50, 100]
    
    for d in dims
        println("\n--- Dimension: $d ---")
        
        # Create test distributions
        μ1 = randn(d)
        μ2 = randn(d)
        
        # Create positive definite covariance matrices
        A1 = randn(d, d)
        Σ1 = A1' * A1 + I(d)
        
        A2 = randn(d, d)
        Σ2 = A2' * A2 + I(d)
        
        left = MvNormal(μ1, Σ1)
        right = MvNormal(μ2, Σ2)
        
        # Warm-up
        compute_logscale_v1(left, right)
        compute_logscale_v2(left, right)
        
        # Benchmark
        println("\nImplementation 1 (cholinv_logdet + dot3arg):")
        b1 = @benchmark compute_logscale_v1($left, $right) samples=1000 evals=10
        display(b1)
        
        println("\nImplementation 2 (logdet + backslash):")
        b2 = @benchmark compute_logscale_v2($left, $right) samples=1000 evals=10
        display(b2)
        
        # Compare results
        result1 = compute_logscale_v1(left, right)
        result2 = compute_logscale_v2(left, right)
        
        println("\nResults comparison:")
        println("  Implementation 1: $result1")
        println("  Implementation 2: $result2")
        println("  Difference: $(abs(result1 - result2))")
        
        # Speed comparison
        ratio = median(b2).time / median(b1).time
        if ratio > 1
            println("\n✓ Implementation 1 is in average $(round(ratio, digits=2))x faster")
        else
            println("\n✓ Implementation 2 is in average $(round(1/ratio, digits=2))x faster")
        end
    end
    
    println("\n" * "=" ^ 70)
end

# Run the benchmark
benchmark_compute_logscale_normalxnormal()
