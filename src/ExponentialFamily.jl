module ExponentialFamily
using TinyHugeNumbers

# Reexport `tiny` and `huge` from the `TinyHugeNumbers`
export tiny, huge

include("constants.jl")
include("prod.jl")
include("distributions.jl")
include("exponential_family.jl")
# include("cached.jl")

include("helpers/fixes.jl")
include("helpers/algebra/cholesky.jl")
include("helpers/algebra/common.jl")
include("helpers/algebra/combinatorics.jl")
include("helpers/algebra/correction.jl")

include("distributions/bernoulli.jl")
include("distributions/categorical.jl")
include("distributions/gamma_family/gamma_shape_rate.jl")
include("distributions/gamma_family/gamma_shape_scale.jl")
include("distributions/gamma_family/gamma_family.jl")
include("distributions/normal_family/normal_mean_variance.jl")
include("distributions/normal_family/normal_mean_precision.jl")
include("distributions/normal_family/normal_weighted_mean_precision.jl")
include("distributions/normal_family/mv_normal_mean_covariance.jl")
include("distributions/normal_family/mv_normal_mean_precision.jl")
include("distributions/normal_family/mv_normal_weighted_mean_precision.jl")
include("distributions/normal_family/normal_family.jl")
include("distributions/gamma_inverse.jl")
include("distributions/geometric.jl")
include("distributions/matrix_dirichlet.jl")
include("distributions/dirichlet.jl")
include("distributions/beta.jl")
include("distributions/lognormal.jl")
include("distributions/binomial.jl")
# include("distributions/multinomial.jl")
# include("distributions/wishart.jl")
# include("distributions/wishart_inverse.jl")
# include("distributions/contingency.jl")
include("distributions/erlang.jl")
include("distributions/exponential.jl")
include("distributions/von_mises_fisher.jl")
include("distributions/von_mises.jl")
include("distributions/pareto.jl")
# include("distributions/continuous_bernoulli.jl")
# include("distributions/negative_binomial.jl")
include("distributions/rayleigh.jl")
include("distributions/weibull.jl")
include("distributions/laplace.jl")
include("distributions/poisson.jl")
include("distributions/chi_squared.jl")
# include("distributions/mv_normal_wishart.jl")
include("distributions/normal_gamma.jl")

end
