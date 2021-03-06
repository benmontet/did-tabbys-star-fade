import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import emcee
import corner


def lnlike(theta,  x,  y, yerr):
    m, b, lnf = theta
    model = m * x + b
    inv_sigma2 = 1.0/(yerr**2 + model**2*np.exp(2*lnf))
    return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))


def lnprior(theta):
    m, b, lnf = theta
    if -1. < m < 1. and -1.E3 < b < 1.E4 and -10.0 < lnf < 10.0:
        return 0.0
    return -np.inf


def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)


def leastsq(x, y, yerr, print_output=False, plot_output=False):
    A = np.vstack((np.ones_like(x), x)).T
    C = np.diag(yerr * yerr)
    cov = np.linalg.inv(np.dot(A.T, np.linalg.solve(C, A)))
    b_ls, m_ls = np.dot(cov, np.dot(A.T, np.linalg.solve(C, y)))

    if print_output:
        print("""Least-squares results:
            m = {0} ± {1}
            b = {2} ± {3}
            """.format(m_ls, np.sqrt(cov[1, 1]), b_ls, np.sqrt(cov[0, 0])))

    if plot_output:
        fig, ax1 = plt.subplots(1, 1, figsize=[8, 5])
        ax1.errorbar(x, y, yerr=yerr, fmt=".k")
        ax1.plot(xl, m_ls*xl+b_ls, "--k")
        ax1.set_xlim(xmin, xmax)
        ax1.set_xlabel("$x$")
        ax1.set_ylabel("$y$")
        ax1.set_ylim(ymin, ymax)
        ax1.set_title('Least-squares fit')
        ax1.tight_layout()

        return b_ls, m_ls, cov, fig

    return b_ls, m_ls, cov


def maxlike(x, y, yerr, print_output=False, plot_output=False):
    nll = lambda *args: -lnlike(*args)
    result = op.minimize(nll, [0.01, 12, 0.01], args=(x, y, yerr))
    m_ml, b_ml, lnf_ml = result["x"]
    if print_output:
        print("""Max like results:
            m = {0}
            b = {2}
            """.format(m_ml, b_ml))

    if plot_output:
        fig, ax1 = plt.subplots(1, 1, figsize=[8, 5])
        ax1.errorbar(x, y, yerr=yerr, fmt=".k")
        ax1.plot(xl, m_ml*xl+b_ml, "--k") 
        ax1.set_xlim(xmin, xmax)
        ax1.set_xlabel("$x$")
        ax1.set_ylabel("$y$")
        ax1.set_ylim(ymin, ymax)
        ax1.set_title('Least-squares fit')
        ax1.tight_layout()

        return m_ml, b_ml, lnf_ml, fig

    return m_ml, b_ml, lnf_ml


def mcmc(x, y, yerr, print_output=False,
         ndim=3, nwalkers=300, nsamp=1500,
         burnin=100):

    m_ml, b_ml, lnf_ml = maxlike(x, y, yerr, print_output=False)
    if not np.isfinite(lnprob([m_ml, b_ml, lnf_ml], x, y, yerr)):
        m_ml, b_ml, lnf_ml = 0.0016, 9.25, np.log(0.0013)

    pos = [np.array([m_ml, b_ml, lnf_ml]) +
        1e-3*np.random.randn(ndim) for i in range(nwalkers)]

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, lnprob, args=(x, y, yerr))
    _ = sampler.run_mcmc(pos, 5000)

    samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))

    samples[:, 2] = np.exp(samples[:, 2])
    m_mcmc, b_mcmc, f_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                                 zip(*np.percentile(samples, [16, 50, 84],
                                                    axis=0)))
    if print_output:
        print("""MCMC result:
            m = {0[0]} +{0[1]} -{0[2]}
            b = {1[0]} +{1[1]} -{1[2]}
            f = {2[0]} +{2[1]} -{2[2]}
            """.format(m_mcmc, b_mcmc, f_mcmc))

    return sampler


def plot_chains(sampler):
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 15))
    axes[0].plot(sampler.chain[:, :, 0].T, color="k", alpha=0.4)
    axes[0].set_ylabel("$m$")
    axes[1].plot(sampler.chain[:, :, 1].T, color="k", alpha=0.4)
    axes[1].set_ylabel("$b$")
    axes[2].plot(np.exp(sampler.chain[:, :, 2]).T, color="k", alpha=0.4)
    axes[2].set_ylabel("$f$")
    axes[2].set_xlabel("step number")
    for ax in axes:
        ax.minorticks_on()
        ax.grid()
    fig.tight_layout(h_pad=0.0)
    return fig


def plot_corner(sampler, burnin=100, ndim=3):
    samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
    fig = corner.corner(samples, labels=["$m$", "$b$", "$\ln\,f$"], )
    for ax in fig.axes:
        ax.minorticks_on()
        ax.grid()
    return fig

def plot_samples(sampler, data, sampsize=30000, burnin=100, ndim=3,
        fill=True,):
    x, y, yerr = data
    size = sampsize
    samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
    xval = np.arange(xmin, xmax, 1)
    savearr = np.zeros([size, len(xval)])
    for i, [m, b, lnf] in enumerate(samples[
            np.random.randint(len(samples), size=size)]):
        savearr[i] = m*xval+b

    pc = np.percentile(savearr, [16, 50, 84], axis=0)

    fig, ax1 = plt.subplots(1, 1, figsize=[8, 5])
    if fill:
        ax1.fill_between(xval, pc[0], pc[2], alpha=0.5)
    else:
        for m, b, lnf in samples[np.random.randint(len(samples), size=100)]:
            ax1.plot(xval, m*xval+b, color="k", alpha=0.05)
    ax1.errorbar(x, y, yerr=yerr, fmt=".k")
    ax1.minorticks_on()
    ax1.grid()
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(ymin, ymax)
    ax1.set_xlabel('Year')
    ax1.set_ylabel('$B$ ')
    #ax1.text(xmax, 12.515, 'Figure by Tom Barclay / @mrtommyb',
    #    fontsize=8, ha='right')
    fig.tight_layout()
    return fig

# some constants
xmin = 1882
xmax = 1993
ymin = 12.48
ymax = 12.22
xl = [xmin,xmax]

