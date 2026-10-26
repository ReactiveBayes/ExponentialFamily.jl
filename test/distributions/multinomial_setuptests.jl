include("distributions_setuptests.jl")

# Enumerate small supports independently of the production convolution.
function multinomial_test_support(n, categories)
    return [[x..., n - sum(x; init = 0)] for x in Iterators.product(ntuple(_ -> 0:n, categories - 1)...) if sum(x; init = 0) <= n]
end
