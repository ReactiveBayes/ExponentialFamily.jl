module ExponentialFamilyTests

using ExponentialFamily, Distributions, FastCholesky
using Test

include("testutils.jl")

@testset "ExponentialFamily" begin
    include("test_prod.jl")
    include("test_distributions.jl")
    include("test_exponential_family.jl")

    include("distributions/test_bernoulli.jl")
    include("distributions/test_categorical.jl")
    include("distributions/normal_family/test_mv_normal_mean_covariance.jl")
    include("distributions/normal_family/test_mv_normal_mean_precision.jl")
    include("distributions/normal_family/test_mv_normal_weighted_mean_precision.jl")
    include("distributions/normal_family/test_normal_mean_variance.jl")
    include("distributions/normal_family/test_normal_mean_precision.jl")
    include("distributions/normal_family/test_normal_weighted_mean_precision.jl")
    include("distributions/normal_family/test_normal_family.jl")
    include("distributions/gamma_family/test_gamma_shape_rate.jl")
    include("distributions/gamma_family/test_gamma_shape_scale.jl")
    include("distributions/gamma_family/test_gamma_family.jl")
    include("distributions/test_binomial.jl")
    include("distributions/test_beta.jl")
    # include("distributions/test_contingency.jl")
    include("distributions/test_matrix_dirichlet.jl")
    include("distributions/test_dirichlet.jl")
    include("distributions/test_exponential.jl")
    include("distributions/test_gamma_inverse.jl")
    include("distributions/test_lognormal.jl")
    # include("distributions/test_multinomial.jl")
    include("distributions/test_geometric.jl")
    include("distributions/test_poisson.jl")
    include("distributions/test_wishart.jl")
    include("distributions/test_wishart_inverse.jl")
    include("distributions/test_erlang.jl")
    include("distributions/test_von_mises_fisher.jl")
    include("distributions/test_vonmises.jl")
    include("distributions/test_pareto.jl")
    # include("distributions/test_continuous_bernoulli.jl")
    include("distributions/test_negative_binomial.jl")
    include("distributions/test_rayleigh.jl")
    include("distributions/test_weibull.jl")
    include("distributions/test_laplace.jl")
    include("distributions/test_chi_squared.jl")
    include("distributions/test_mv_normal_wishart.jl")
    include("distributions/test_normal_gamma.jl")
end

end
