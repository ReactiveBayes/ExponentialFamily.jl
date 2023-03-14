module ExponentialFamily
using TinyHugeNumbers

# Reexport `tiny` and `huge` from the `TinyHugeNumbers`
export tiny, huge
include("plus_traits.jl")
include("distributions.jl")
include("natural_parameters.jl")

include("helpers/fixes.jl")
include("helpers/algebra/cholesky.jl")
include("helpers/algebra/common.jl")
include("helpers/algebra/correction.jl")
include("distributions/gamma_shape_rate.jl")
include("distributions/gamma.jl")
include("distributions/gamma_inverse.jl")
include("distributions/categorical.jl")
include("distributions/matrix_dirichlet.jl")
include("distributions/dirichlet.jl")
include("distributions/beta.jl")
include("distributions/bernoulli.jl")
include("distributions/normal_mean_variance.jl")
include("distributions/normal_mean_precision.jl")
include("distributions/normal_weighted_mean_precision.jl")
include("distributions/mv_normal_mean_covariance.jl")
include("distributions/mv_normal_mean_precision.jl")
include("distributions/mv_normal_weighted_mean_precision.jl")
include("distributions/normal.jl")
include("distributions/wishart.jl")
include("distributions/wishart_inverse.jl")
include("distributions/contingency.jl")
include("distributions/erlang.jl")
include("distributions/exponential.jl")

end
