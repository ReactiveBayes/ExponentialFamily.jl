function monte_carlo_information_matrix(rng, hessian, dist_factory, param, n_samples)
    dist = dist_factory(param)
    samples = rand(rng, dist, n_samples)
    sample_logpdf = (p) -> mean(logpdf.(dist_factory(p), samples))
    return -hessian(sample_logpdf, param)
end
