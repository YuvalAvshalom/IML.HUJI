from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model

    uni = UnivariateGaussian()
    s = np.random.normal(10, 1, 1000)
    res = uni.fit(s)
    print('(' + str(res.mu_) + ', ' + str(res.var_) + ')')

    # Question 2 - Empirically showing sample mean is consistent

    ms = np.linspace(10, 1000, 1000).astype(int)
    mu, sigma = 10, 1
    estimated_mean = []

    for m in ms:
        X = np.random.normal(mu, sigma, size=m)
        estimated_mean.append(uni.fit(X).mu_)

    go.Figure([go.scatter(x=ms, y=estimated_mean, mode='markers+lines')])

    # Question 3 - Plotting Empirical PDF of fitted model


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    raise NotImplementedError()

    # Question 5 - Likelihood evaluation
    raise NotImplementedError()

    # Question 6 - Maximum likelihood
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    # test_multivariate_gaussian()
