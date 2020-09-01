import numpy as np
import matplotlib.pyplot as plt
from pyriemann.utils.distance import (distance_riemann,
                                      distance_euclid)
from pyriemann.utils.mean import (mean_riemann,
                                      mean_euclid)
from pyriemann.estimation import Covariances


def trial_evoked(t, snr=1.0, sigma=0.1, delay=0.3, duration=0.5):
    """Generate a piece of toy signal with a sine wave after a certain
    delay and some noise.
    """
    t_step = np.diff(t).mean()
    t_response = np.arange(0, duration, t_step)
    response = np.sin(t_response * (2 * np.pi / duration))  # looks like evoked
    # response = np.sin(t_response * (2 * np.pi / duration))  # looks like evoked
    # response = np.sin(t_response * (6 * np.pi / duration)) * np.sin(t_response * (np.pi / duration))  # looks like induced?
    # response = np.random.normal(loc=0.0, scale=sigma, size=t_response.size) * np.sin(t_response * (np.pi / duration)) * 3.0 # looks like induced?
    noise = np.random.normal(loc=0.0, scale=sigma, size=t.size)
    x = np.zeros(t.size)
    x[(t >= delay) * (t < delay+duration)] += response
    x = x * snr * sigma + noise
    return x


def trial_induced(t, snr=1.0, sigma=0.1, delay=0.3, duration=0.5,
                  phase=0.0, response_freq=30.0):
    """Generate a piece of toy signal with a sine wave after a certain
    delay and some noise.
    """
    t_step = np.diff(t).mean()
    t_response = np.arange(0, duration, t_step)
    response = np.sin(t * (response_freq * 2.0 * np.pi) + phase) * sigma
    tmp = (t >= delay) * (t < delay+duration)
    response[tmp] *= np.sin(t_response * (np.pi / duration))
    response[~tmp] *= sigma
    noise = np.random.normal(loc=0.0, scale=sigma, size=t.size)
    x = np.zeros(t.size)
    x = response * snr + noise
    return x


def generate_trials(snr, n_trials, t, delay_range, sigma, duration,
                    phase_locked=False, response_freq=30.0):
    n_sensors = len(snr)
    XX = np.zeros((n_trials, n_sensors, t.size))
    for i in range(n_trials):
        delay = np.random.uniform(low=delay_range[0],
                                  high=delay_range[1])
        if phase_locked:
            phase = 0.0
        else:
            phase = np.random.uniform(low=0.0, high=2.0*np.pi)

        for j in range(n_sensors):
            XX[i, j] = trial_induced(t, snr[j], sigma, delay,
                                     duration, phase,
                                     response_freq=response_freq)

    return XX


def plot_trials(XX, t, max_subplots=8, linewidth=3):
    n_trials = XX.shape[0]
    n_sensors = XX.shape[1]
    fig, axs = plt.subplots(max_subplots, 1,
                            sharex=True, sharey=True)
    fig.set_figheight(10)
    fig.set_figwidth(6)
    fig.set_dpi(100)
    fig.set_facecolor('w')
    fig.set_edgecolor('k')
    offset = np.arange(n_sensors)
    color = ['k'] * n_sensors
    signal_max = 0.0
    color = ['k'] * n_sensors
    signal_max = 0.0
    for i in range(n_trials):
        for j in range(n_sensors):
            if i < max_subplots - 1:
                axs[i].plot(t, XX[i, j] + offset[j], color[j])
                if np.abs(XX[i, j]).max() > signal_max:
                    signal_max = np.abs(XX[i, j]).max()

            else:
                continue

        plt.ylim([offset.min() - signal_max, offset.max() + signal_max])
        plt.gca().invert_yaxis()

    for j in range(n_sensors):
        axs[max_subplots-1].plot(t, XX[:, j].mean(0) + offset[j], color[j],
                                 linewidth=linewidth)
        plt.gca().invert_yaxis()

    plt.yticks(range(n_sensors))
    # plt.axis('tight')
    fig.tight_layout()
    return XX


def plot_trials_erf(snr, n_trials, t, delay_range, sigma, duration,
                    category, max_subplots=8, phase_locked=False,
                    response_freq=30.0, linewidth=3):
    """Generate simulated trials from a set of sensors.
    """
    n_sensors = len(snr)
    XX = np.zeros((n_trials, n_sensors, t.size))
    fig, axs = plt.subplots(max_subplots, 1,
                            sharex=True, sharey=True)
    fig.set_figheight(10)
    fig.set_figwidth(6)
    fig.set_dpi(100)
    fig.set_facecolor('w')
    fig.set_edgecolor('k')
    offset = np.arange(n_sensors)
    # fig.suptitle("Trials and ERF - Category %s" % category)
    # color = ['b', 'g', 'k']
    color = ['k'] * n_sensors
    signal_max = 0.0
    for i in range(n_trials):
        delay = np.random.uniform(low=delay_range[0],
                                  high=delay_range[1])
        if phase_locked:
            phase = 0.0
        else:
            phase = np.random.uniform(low=0.0, high=2.0*np.pi)

        for j in range(n_sensors):
            XX[i, j] = trial_induced(t, snr[j], sigma, delay,
                                     duration, phase, response_freq=response_freq)
            if i < max_subplots - 1:
                axs[i].plot(t, XX[i, j] + offset[j], color[j])
                if np.abs(XX[i, j]).max() > signal_max:
                    signal_max = np.abs(XX[i, j]).max()

            else:
                continue

        plt.ylim([offset.min() - signal_max, offset.max() + signal_max])
        plt.gca().invert_yaxis()

    for j in range(n_sensors):
        axs[max_subplots-1].plot(t, XX[:, j].mean(0) + offset[j], color[j],
                                 linewidth=linewidth)
        plt.gca().invert_yaxis()

    plt.yticks(range(n_sensors))
    # plt.axis('tight')
    fig.tight_layout()
    return XX


def plot_cov(C, title, vmin=None, vmax=None):
    fig = plt.figure()
    fig.set_figheight(5)
    fig.set_figwidth(5)
    fig.set_dpi(100)
    fig.set_facecolor('w')
    fig.set_edgecolor('k')
    plt.title(title)

    plt.imshow(C, interpolation='nearest', cmap='jet',
               vmin=vmin, vmax=vmax)
    plt.xticks(range(C.shape[1]))
    plt.yticks(range(C.shape[1]))
    plt.colorbar()
    fig.tight_layout()
    return fig


def plot_covs(XX, category, normalize=False, train_size=0.7,
              estimator='scm'):
    if normalize:
        XX = (XX - XX.mean(2)[:, :, None]) / XX.std(2)[:, :, None]

    n_trials = XX.shape[0]
    # covs = np.array([np.cov(XX[i]) for i in range(n_trials)])
    covs = Covariances(estimator=estimator).fit_transform(XX)
    vmin = None
    vmax = None
    if normalize:
        vmin = 0.0
        vmax = 1.0

    title = "Mean Covariance - Category %s" % category
    covs_mean = covs.mean(0)
    plot_cov(covs_mean, title, vmin, vmax)
    return covs


def my_var(cs, mean_c):
    """Variance of a set cs of covariances given the mean covariance
    mean_c.
    """
    return np.power(cs - mean_c, 2).sum(0)


def plot_mdm(d1, d2, y, w_guess, b_guess, label, save=False, outdir=None, figure_format='pdf'):
    fig = plt.figure()
    fig.set_figheight(6)
    fig.set_figwidth(8)
    fig.set_dpi(100)
    fig.set_facecolor('w')
    fig.set_edgecolor('k')
    plt.plot(d1[y == 1], d2[y == 1], 'ro', label='1')
    plt.plot(d1[y == 0], d2[y == 0], 'bo', label='2')
    # plot linear discrimination function:
    plt.axis('auto')
    ax = plt.gca()
    x_vals = np.array(ax.get_xlim())
    y_vals = -(x_vals * w_guess[0, 0] + b_guess)/w_guess[0, 1]
    plt.plot(x_vals, y_vals, '-', c="k")
    if label == 'trial':
        variable = 'X'
    else:
        variable = 'C'

    plt.xlabel("%s distance from $\overline{%s}_{1}$" % (label, variable))
    plt.ylabel("%s distance from $\overline{%s}_{2}$" % (label, variable))
    # plt.legend(loc='upper left')
    plt.legend(numpoints=1)
    fig.tight_layout()
    if save:
        filename = outdir + '%s_distance.%s' % ('_'.join(label.split()), figure_format)
        print("Saving figure to %s" % filename)
        plt.savefig(filename)

    return
