+++
title = "An iterative exploration on EM algorithm"
date = 2024-01-02T23:00:05+08:00
tags = ["machine learning", "math"]
menu = "main"
+++

# An iterative exploration on EM algorithm

As a continuation of my notes on classicial machine learning, this is an
exclusive study on EM agorithm to deepen my understanding from a wider
variety of perspectives.

## EM as Expection + Maximization

The understanding of latent variable is the first \`get\` in EM
algorithms.

Through latent variable, we have witnessed the greatness of EM algorithm
which breaks through the limit of MLE.

Likelihood of latent variables:

{{< katex >}}
L(\theta;{\bf X})=p({\bf X} | \theta)=\int p({\bf X},{\bf Z} | \theta)d{\bf Z}
{{< /katex >}}

E-step:

{{< katex >}}
Q(\theta|\theta^{(i)})=\operatorname{E}_{\mathbf{Z},\mathbf{X},\theta^{(i)}}\left[\log L(\theta;\mathbf{X},\mathbf{Z})\right]
{{< /katex >}}

M-step:

{{< katex >}}
\theta^{(t+1)} = \underset{\theta}{\arg\max} \, Q(\theta|\theta^{(t)})
{{< /katex >}}

Here's the same example using a Gaussian Mixture Model (GMM) with two
components in Python, using NumPy and SciPy:

1. Import required libraries and generate data:

``` python
import numpy as np
from scipy.stats import norm

np.random.seed(42)
data = np.concatenate((np.random.normal(1, 1, 100), np.random.normal(5, 1, 100)))
```

``` python
print(len(data))
print(data[:5])
```

2. Define the log-likelihood function for the complete data (X, Z):

``` python
def log_likelihood(data, means, pis, k):
    return np.sum(np.log(np.sum(pis[j] * norm.pdf(data, means[j], 1) for j in range(k))))
```

3. Implement the EM algorithm:

``` python
def EM(data, k, max_iterations):
    means = np.random.uniform(np.min(data), np.max(data), k)
    pis = np.full(k, 1/k)

    for _ in range(max_iterations):
        # E-step
        assignments = np.array([[pis[j] * norm.pdf(x, means[j], 1) for j in range(k)] for x in data])
        assignments = assignments / np.sum(assignments, axis=1).reshape(-1, 1)

        # M-step
        means = np.sum(assignments * data.reshape(-1, 1), axis=0) / np.sum(assignments, axis=0)
        pis = np.sum(assignments, axis=0) / len(data)

    return means, pis
```

4. Run the EM algorithm with the generated data, 2 components, and 50
    iterations:

``` python
means, pis = EM(data, 2, 50)
```

5. Visualize the result:

``` python
import matplotlib.pyplot as plt

plt.hist(data, bins=20, density=True, alpha=0.6)
x = np.linspace(np.min(data), np.max(data), 1000)
gmm_pdf = pis[0] * norm.pdf(x, means[0], 1) + pis[1] * norm.pdf(x, means[1], 1)
plt.plot(x, gmm_pdf, 'r-', linewidth=2)
plt.savefig("./gmm_pdf.png")
plt.show()
```

![](/img/gmm_pdf.png)

## EM as a local lower bound construction

If you delve deeper into the convergence proof of the EM algorithm based
on latent variables, using the *Jensen's inequality* construction for
the log(x) function [^1], we can easily prove that the EM algorithm
**repeatedly constructs new lower bounds and then further solves them**.

Thus the EM process can be seen as: fix the current parameters
{{< katex >}}\theta_n{{< /katex >}} first, calculate a lower bound function for the distribution
of the latent variables, optimize this function to obtain new
parameters, and then repeat.

The EM process described here is connected to the [Variational
Bayes](https://en.wikipedia.org/wiki/Variational_Bayesian_methods) (VB)
method:

> Variational Bayes can be seen as an extension of the
> expectation-maximization (EM) algorithm from maximum a posteriori
> estimation (MAP estimation) of the single most probable value of each
> parameter to fully Bayesian estimation which computes (an
> approximation to) the entire posterior distribution of the parameters
> and latent variables. As in EM, it finds a set of optimal parameter
> values, and it has the same alternating structure as does EM, based on
> a set of interlocked (mutually dependent) equations that cannot be
> solved analytically.

Both methods involve finding approximate solutions to intractable
optimization problems by constructing lower bounds for the objective
functions. Similarly, in the Variational Bayes method, the goal is to
approximate the posterior distribution of latent variables by minimizing
the Kullback-Leibler (KL) divergence between the true posterior
distribution and the approximate distribution. This is achieved by
constructing a lower-bound function for the log marginal likelihood,
called the Evidence Lower Bound (ELBO), and optimizing this lower-bound
function with respect to the approximate distribution.

## K-means as a hard EM

Based on the understanding of the second level, you can now freely apply
the EM algorithm to GMM and HMM models. Especially after a deep
understanding of GMM, for the joint probability with latent variables,
when using the Gaussian distribution as a substitute:

{{< katex >}}\begin{aligned} P_{\Theta}\left(x_1, \ldots, x_n, z_1, \ldots z_n\right) & =\prod_{t=1}^N P_{\Theta}\left(z_t\right) P_{\Theta}\left(x_t \mid z_t\right) \\ & =\prod_{t=1}^N \frac{1}{K} \mathcal{N}\left(\mu^{z_t}, I\right)\left(x_t\right)\end{aligned}{{< /katex >}}

This formula represents the joint probability distribution for a set of
observed variables {{< katex >}}x_1, \ldots, x_n{{< /katex >}} and latent variables
{{< katex >}}z_1, \ldots, z_n{{< /katex >}} under a Gaussian Mixture Model (GMM) with parameter
{{< katex >}}\Theta{{< /katex >}}.

The formula can be broken down as follows:

1.  {{< katex >}}P_{\Theta}\left(x_1, \ldots, x_n, z_1, \ldots z_n\right){{< /katex >}}: This is
    the joint probability of the observed variables {{< katex >}}x_1, \ldots, x_n{{< /katex >}}
    and the latent variables {{< katex >}}z_1, \ldots, z_n{{< /katex >}} under parameter
    {{< katex >}}\Theta{{< /katex >}}.
2.  {{< katex >}}\prod_{t=1}^N P_{\Theta}\left(z_t\right) P_{\Theta}\left(x_t \mid z_t\right){{< /katex >}}:
    This expression is the product of the prior probabilities of the
    latent variables {{< katex >}}z_t{{< /katex >}} and the conditional probabilities of the
    observed variables {{< katex >}}x_t{{< /katex >}} given the latent variables {{< katex >}}z_t{{< /katex >}}. The
    product is taken over all {{< katex >}}N{{< /katex >}} data points.
3.  {{< katex >}}\frac{1}{K}{{< /katex >}}: This term represents the uniform prior distribution
    of the latent variables {{< katex >}}z_t{{< /katex >}}, where {{< katex >}}K{{< /katex >}} is the number of Gaussian
    components in the GMM.
4.  {{< katex >}}\mathcal{N}\left(\mu^{z_t}, I\right)\left(x_t\right){{< /katex >}}: This term is
    the conditional probability of the observed variable {{< katex >}}x_t{{< /katex >}} given the
    latent variable {{< katex >}}z_t{{< /katex >}}. It is modeled as a Gaussian distribution with
    mean {{< katex >}}\mu^{z_t}{{< /katex >}} and identity covariance matrix {{< katex >}}I{{< /katex >}}.

Easy to see a connection with MSE:

{{< katex >}}\left(\mu^1, \ldots, \mu^K\right)^*=\underset{\mu^1, \ldots, \mu^k}{\operatorname{argmin}} \underset{z_1, \ldots, z_n}{\min}\,\sum\limits_{t=1}^N\left\|\mu^{z_t}-x_t\right\|^2{{< /katex >}}

This formula represents the optimization problem for finding the optimal
means {{< katex >}}\left(\mu^1, \ldots, \mu^K\right)^*{{< /katex >}} of a Gaussian Mixture Model
(GMM) by minimizing the mean squared distance between the data points
and their corresponding means.

The formula can be broken down as follows:

1.  {{< katex >}}\left(\mu^1, \ldots, \mu^K\right)^*{{< /katex >}}: This represents the optimal
    means of the Gaussian components in the GMM.
2.  {{< katex >}}\underset{\mu^1, \ldots, \mu^k}{\operatorname{argmin}}{{< /katex >}}: This
    indicates that we are looking for the values of
    {{< katex >}}\mu^1, \ldots, \mu^k{{< /katex >}} that minimize the expression that follows.
3.  {{< katex >}}\underset{z_1, \ldots, z_n}{\min}{{< /katex >}}: This indicates that we are
    looking for the values of the latent variables {{< katex >}}z_1, \ldots, z_n{{< /katex >}}
    that minimize the expression that follows.
4.  {{< katex >}}\sum\limits_{t=1}^N\left\|\mu^{z_t}-x_t\right\|^2{{< /katex >}}: This is the sum
    of the squared Euclidean distances between each data point {{< katex >}}x_t{{< /katex >}} and
    its corresponding mean {{< katex >}}\mu^{z_t}{{< /katex >}}. The sum is taken over all {{< katex >}}N{{< /katex >}}
    data points.

A simpler and more intuitive explanation is that the K-means algorithm
uses a [hard
clustering](https://learn.microsoft.com/en-us/analysis-services/data-mining/microsoft-clustering-algorithm-technical-reference?view=asallproducts-allversions)
algorithm, while the EM algorithm we are discussing is a soft clustering
algorithm. The so-called hard is a binary decision, either it is or it
isn't (0-1 choice). On the other hand, the soft deals with situations
like a data point always belongs to multiple clusters, and that a
probability is calculated for each combination of data point and
cluster.

This can be [summarized](https://stats.stackexchange.com/a/78278) as:

> There is no "k-means algorithm". There is MacQueens algorithm for
> k-means, the Lloyd/Forgy algorithm for k-means, the Hartigan-Wong
> method, …
>
> There also isn't "the" EM-algorithm. It is a general scheme of
> repeatedly expecting the likelihoods and then maximizing the model.
> The most popular variant of EM is also known as "Gaussian Mixture
> Modeling" (GMM), where the model are multivariate Gaussian
> distributions.
>
> One can consider Lloyds algorithm to consist of two steps:
>
> - the E-step, where each object is assigned to the centroid such that
>   it is assigned to the most likely cluster.
> - the M-step, where the model (centroids) are recomputed (least
>   squares optimization).
>
> … iterating these two steps, as done by Lloyd, makes this effectively
> an instance of the general EM scheme. It differs from GMM that:
>
> - it uses hard partitioning, i.e. each object is assigned to exactly
>   one cluster
> - the model are centroids only, no covariances or variances are taken
>   into account

## EM as a special case of generalized EM

We define the right side of Jensen's inequality as the [free
energy](https://en.wikipedia.org/wiki/Thermodynamic_free_energy).

{{< katex >}}
\mathcal{F}(q, \theta)=\langle\log P(\mathcal{X}, \mathcal{Y} \mid \theta)\rangle_{q(\mathcal{X})}+\mathbf{H}[q]
{{< /katex >}}

This formula represents the free energy {{< katex >}}\mathcal{F}(q, \theta){{< /katex >}}, which
is an important concept in variational inference and is used to
approximate the log marginal likelihood. The formula consists of two
parts:

1.  {{< katex >}}\langle\log P(\mathcal{X}, \mathcal{Y} \mid \theta)\rangle_{q(\mathcal{X})}{{< /katex >}}:
    This term represents the expected log joint probability of the
    observed data {{< katex >}}\mathcal{Y}{{< /katex >}} and the latent variables {{< katex >}}\mathcal{X}{{< /katex >}}
    given the model parameters {{< katex >}}\theta{{< /katex >}}. The expectation is taken with
    respect to the approximate posterior distribution {{< katex >}}q(\mathcal{X}){{< /katex >}}.
    This term measures the goodness-of-fit of the model to the data,
    taking into account the uncertainty in the latent variables.

2.  {{< katex >}}\mathbf{H}[q]{{< /katex >}}: This term represents the entropy of the approximate
    posterior distribution {{< katex >}}q(\mathcal{X}){{< /katex >}}. It measures the uncertainty
    in the latent variables given the observed data. High entropy
    corresponds to a more dispersed distribution, while low entropy
    corresponds to a more concentrated distribution.

The free energy combines these two terms, balancing the goodness-of-fit
of the model with the uncertainty in the latent variables. In
variational inference, the goal is to minimize the free energy with
respect to both the approximate posterior distribution {{< katex >}}q(\mathcal{X}){{< /katex >}}
and the model parameters {{< katex >}}\theta{{< /katex >}}. This minimization leads to an
approximation of the true posterior distribution of the latent variables
and provides a lower bound on the log marginal likelihood.

Thus, **E-step is to optimize the latent (approximate posterior)
distribution {{< katex >}}q(\mathcal{X}){{< /katex >}} with fixed model parameters {{< katex >}}\theta{{< /katex >}}, and
the M-step is to optimize {{< katex >}}\theta{{< /katex >}} with a fixed {{< katex >}}q(\mathcal{X}){{< /katex >}}. This
is the generalized EM algorithm**.

E-Step :

{{< katex >}}
q^{(k)}(\mathcal{X}) :=\underset{q(\mathcal{X})}{\operatorname{argmax}} \mathcal{F}\left(q(\mathcal{X}), \theta^{(k-1)}\right)
{{< /katex >}}

M-step:

{{< katex >}}
\theta^{(k)} :=\underset{\theta}{\operatorname{argmax}} \mathcal{F}\left(q^{(k)}(\mathcal{X}), \theta\right)
{{< /katex >}}

After understanding the generalized EM algorithm, we delve deeper into
free energy and discover the relationship between free energy,
likelihood, and KL divergence. When the model parameters are fixed, the
only option is to optimize the KL divergence. In this case, the hidden
distribution can only take the following form:

{{< katex >}}q^{(k)}(\mathcal{X})=P\left(\mathcal{X} \mid \mathcal{Y}, \theta^{(k-1)}\right){{< /katex >}}

In the EM algorithm, **this is directly given**. Therefore, the EM
algorithm is a naturally optimal hidden distribution case within the
generalized EM algorithm. **However, in many cases, the hidden
distribution is not so easy to compute...**

One example where the hidden distribution is not easy to compute arises
in the context of topic models, such as Latent Dirichlet Allocation
(LDA). In LDA, the goal is to learn the hidden topics that generate a
collection of documents. The observed data are the words in each
document, and the latent variables are the topic assignments for each
word. The hidden distribution in this case is the posterior distribution
of the topic assignments given the observed words and the model
parameters (topic-word probabilities and document-topic probabilities).

Computing the exact hidden distribution in LDA is challenging because
the posterior distribution involves a large number of topic assignments,
which grow exponentially with the number of words and topics. This makes
exact inference intractable for all but the smallest datasets and
simplest models.

To overcome this difficulty, various approximate inference techniques
are employed in practice to estimate the hidden distribution in LDA,
such as:

1.  Gibbs sampling: A Markov chain Monte Carlo (MCMC) method that
    generates samples from the posterior distribution by iteratively
    sampling topic assignments for each word in the documents.
2.  Variational inference: A deterministic method that approximates the
    true posterior distribution with a simpler distribution (e.g., a
    factorized distribution) and minimizes the KL divergence between the
    true and approximate distributions.
3.  Collapsed variational inference or collapsed Gibbs sampling:
    Techniques that integrate out some of the model parameters (e.g.,
    topic-word probabilities) to simplify the inference problem and
    reduce the computational complexity.

[^1]: (Jensen's inequality) Let {{< katex >}}f{{< /katex >}} be a [convex
    function](https://en.wikipedia.org/wiki/Convex_function) on interval
    {{< katex >}}I{{< /katex >}}: If {{< katex >}}x_{1}, x_{2}, \ldots, x_{n} \in I{{< /katex >}} and
    {{< katex >}}\lambda_{1}, \lambda_{2}, \ldots, \lambda_{n} \geq 0{{< /katex >}} with
    {{< katex >}}\sum_{i=1}^{n} \lambda_{i}=1{{< /katex >}}, we have:

    {{< katex >}}
    f\left(\sum_{i=1}^{n} \lambda_{i} x_{i}\right) \leq \sum_{i=1}^{n} \lambda_{i} f\left(x_{i}\right)
    {{< /katex >}}

    Since {{< katex >}}\ln (x){{< /katex >}} is concave (negative convex), we may apply Jensen's
    inequality:

    {{< katex >}}
    \ln \sum_{i=1}^{n} \lambda_{i} x_{i} \geq \sum_{i=1}^{n} \lambda_{i} \ln \left(x_{i}\right)
    {{< /katex >}}

    This result enables us to repeatedly establish a lower bound for the
    logarithm of a sum, which is a key step in deriving the EM
    algorithm.
