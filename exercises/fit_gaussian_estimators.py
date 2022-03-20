from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"

SAMPLES_NUM = 1000


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model

    uni = UnivariateGaussian()
    mu, sigma = 10, 1
    s = np.random.normal(mu, sigma, SAMPLES_NUM)
    res = uni.fit(s)
    print('(' + str(res.mu_) + ', ' + str(res.var_) + ')')

    # Question 2 - Empirically showing sample mean is consistent

    ms = np.linspace(10, 1000, 100).astype(int)
    diff = []

    for m in ms:
        diff.append(abs(uni.fit(s[0:m]).mu_ - mu))

    go.Figure([go.Scatter(x=ms, y=diff, mode='markers+lines')],
              layout=go.Layout(title=r"$\text{ Distance between estimated - "
                                     r"and true value of the expectation as a function of samples number}$",
                               xaxis_title="$m\\text{ - number of samples}$",
                               yaxis_title="r$distance$",
                               height=500)).show()

    # Question 3 - Plotting Empirical PDF of fitted model

    pdf_values = uni.pdf(s)

    go.Figure([go.Scatter(x=s, y=pdf_values, mode='markers')],
              layout=go.Layout(title=r"$\text{ Sampled values distribution}$",
                               xaxis_title="$m\\text{ - sampled values}$",
                               yaxis_title="r$ pdf - values$",
                               height=500)).show()

    # As I expected, the samples' distribution is gaussian around the expectation (10)


def test_multivariate_gaussian():

    # Question 4 - Draw samples and print fitted model

    multi_uni = MultivariateGaussian()
    mu = np.array([0, 0, 4, 0])
    sigma = np.asarray([[1, 0.2, 0, 0.5],
                      [0.2, 2, 0, 0],
                      [0, 0, 1, 0],
                      [0.5, 0, 0, 1]])
    s = np.random.multivariate_normal(mu, sigma, SAMPLES_NUM)
    res = multi_uni.fit(s)
    print(str(res.mu_) + '\n' + str(res.cov_))

    # Question 5 - Likelihood evaluation

    ms = np.linspace(-10, 10, 200)
    logs = np.zeros((200, 200))
    i = 0
    j = 0
    for f1 in ms:
        for f3 in ms:
            logs[i][j] = (MultivariateGaussian.log_likelihood(np.transpose([f1, 0, f3, 0]), sigma, s))
            j += 1
        j = 0
        i += 1

    go.Figure([go.Heatmap(x=ms, y=ms, z=np.asarray(logs))], layout=go.Layout(title=
                                                                 r"$\text{ Log Likelihood as function of "
                                                                 r"different expectancies}$",
                                                                 width=700, height=700,
                                                                 xaxis_title="$f3$", yaxis_title="$f1$")).show()

    # Question 6 - Maximum likelihood

    index = np.argmax(logs)
    row = int(index / 200)
    col = int(index % 200)
    print("Maximum value is achieved for the pair: f1 = " + str(round(ms[row], 3)) + " f3 = " + str(round(ms[col], 3)))


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()


