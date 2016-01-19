import glob

import scipy.optimize as op
import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import emcee

from astropy import log


def lnlike(theta,  x,  y, yerr):
    """Returns the log likelihood."""
    m, b = theta
    model = m * x + b
    inv_sigma2 = 1.0 / (yerr**2)
    return -0.5*(np.sum((y - model)**2 * inv_sigma2 - np.log(inv_sigma2)))


def lnprior(theta):
    """Returns the log prior."""
    m, b = theta
    if -1. < m < 1. and -100.0 < b < 100.0:
        return 0.0  # log(Probability=1) = 0
    return -np.inf  # log(Probability=verysmall) = -inf


def lnprob(theta, x, y, yerr):
    """Returns the log posterior."""
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)


def mcmc(x, y, yerr, nwalkers=20, nsamp=2000, burnin=100, corner_plot=True):
    """Samples the slope and intercept using MCMC."""
    log.info("Calling EnsembleSampler.run_mcmc")
    # Initialize the model at a good starting position
    ndim = 2
    pos = [np.array([0.01, 12.]) + 1e-4*np.random.randn(ndim)
           for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                    args=(x, y, yerr), threads=20)
    _ = sampler.run_mcmc(pos, 1000)
    samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
    return samples


def run_analysis(df, name="default"):
    """"""
    x = df['decimalyear'].values
    y = df['magcal_magdep'].values
    yerr = df['magcal_local_rms'].values

    samples = mcmc(x, y, yerr)
    samples_m, samples_b = samples[:, 0], samples[:, 1]
    lightcurve_plot(x, y, yerr, samples_m, samples_b,
                    output_fn="output/{}.png".format(name))
    corner_plot(samples, output_fn="output/{}-corner.png".format(name))

    # Summarize the posterior using different statistics
    m_mean, m_std = np.mean(samples_m), np.std(samples_m)
    b_mean, b_std = np.mean(samples_b), np.std(samples_b)
    # Probability that the slope is positive
    prob_slope_positive = (samples_m > 0).sum() / samples_m.size
    # Probability that the slope is negative
    prob_slope_negative = (samples_m < 0).sum() / samples_m.size
    # Probability that the slope is less than 0.05 mag per century
    prob_slope_0p05 = (samples_m < 0.0005).sum() / samples_m.size
    # Write the results to a text file
    out = open("output/mcmc-result-{}.txt".format(name), "w")
    out.write("Results for {}\n=====================\n".format(name))
    out.write("Data points: {}\n".format(len(x)))
    out.write("Period: {:.1f} - {:.1f}\n".format(x.min(), x.max()))
    out.write("MCMC samples: {}\n\n".format(len(samples_m)))

    msg_mcmc = ("Mean and standard deviation of the posterior parameters:\n"
                "m = {:.5f} +/- {:.5f}\n"
                "b = {:.3f} +/- {:.3f}\n\n"
                .format(m_mean, m_std, b_mean, b_std))
    out.write(msg_mcmc)
    msg_verbose = ("i.e. the star changed by:\n"
                   "{0:.3f} +/_ {1:.3f} mag/century\n"
                   "= {0:.2f} +/_ {1:.2f} mag/century\n\n".format(100*m_mean, 100*m_std))
    out.write(msg_verbose)
    out.write("P(slope > 0): {:.1f}%\n".format(100 * prob_slope_positive))
    out.write("P(slope < 0): {:.1f}%\n".format(100 * prob_slope_negative))
    out.write("P(slope < 0.0005): {:.1f}%\n".format(100 * prob_slope_0p05))
    out.close()
    # And write some info to the terminal to update the user
    log.info(msg_mcmc)
    log.info(msg_verbose)
    del samples


def corner_plot(samples, output_fn="corner_plot.png"):
    import corner
    fig = corner.corner(samples, labels=["$m$", "$b$"], )
    log.info("Writing {}".format(output_fn))
    fig.savefig(output_fn)
    pl.close(fig)


def lightcurve_plot(x, y, yerr, samples_m, samples_b, output_fn="plot.png"):
    fig = pl.figure(figsize=(7, 2.5))
    ax = fig.add_subplot(1, 1, 1)
    
    #ax.scatter(x, y, marker="o", s=1, lw=0, facecolor="black")
    #ax.errorbar(x, y, yerr=yerr, fmt=".k", ms=1, capsize=0, alpha=0.3, elinewidth=0.5)
    ax.plot(x, y, ".k", ms=2)

    n_draws = 100  # Number of draws to visualize the posterior in data space
    xval = np.arange(x.min(), x.max(), 1)
    savearr = np.zeros([n_draws, len(xval)])
    for idx, samples_idx in enumerate(np.random.randint(len(samples_m), size=n_draws)):
        savearr[idx] = samples_m[samples_idx] * xval + samples_b[samples_idx]
    pc = np.percentile(savearr, [16, 50, 84], axis=0)
    fill = True
    if fill:
        ax.fill_between(xval, pc[0], pc[2], alpha=0.5)
    else:
        for m, b in samples[np.random.randint(len(samples), size=100)]:
            ax.plot(xl, m * xl + b, color="k", alpha=0.05)
    ax.minorticks_on()
    ax.set_xlim(1885, 1995)
    ax.set_ylim(13., 11.8)
    ax.set_xticks(np.arange(1890, 1991, 10))
    ax.set_xlabel('$\mathrm{Year}$')
    ax.set_ylabel('B')
    fig.tight_layout(pad=0.4)
    fig.savefig(output_fn, dpi=200)
    pl.close(fig)


if __name__ == "__main__":
    df = pd.read_csv("../data/preprocessed-data.csv")

    #run_analysis(df, name="all")

    mask = (df['seriesId'] != 11) & (df['seriesId'] != 12) & (df['seriesId'] != 13)
    run_analysis(df[mask], name="without-series-11-12-13")

    """
    for seriesId in df['seriesId'].unique():
        mask = df['seriesId'] == seriesId
        if mask.sum() >= 20:  # Ignore series with less than 10 points
            run_analysis(df[mask], name="series-{}".format(seriesId))
    """
    """
    for seriesId in df['seriesId'].unique():
        mask = df['seriesId'] != seriesId
        if mask.sum() >= 5:
            run_analysis(df[mask], name="without-series-{}".format(seriesId))
    """
