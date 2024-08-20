import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import norm


pKa1_range = [4, 9]
pKa2_range = [9, 12]
slope_range = [-0.1, 0.0]
E_infinity_range = [-0.6, -0.9]


def value_at_ph(pH, pKa1, pKa2, slope, E_infinity):
    """value_at_ph.

    Parameters
    ----------
    pH :
        pH
    pKa1 :
        pKa1
    pKa2 :
        pKa2
    slope :
        slope
    E_infinity :
        E_infinity
    """
    if pH > pKa2:
        return E_infinity
    if pH > pKa1 and pH < pKa2:
        return E_infinity - (pKa2 - pH) * slope
    return E_infinity - (pKa2 - pKa1) * slope - (pKa1 - pH) * 2 * slope


def gaus_prob(pH, E, pKa1, pKa2, slope, E_infinity, var=0.01):
    """gaus_prob.

    Parameters
    ----------
    pH :
        pH
    E :
        E
    pKa1 :
        pKa1
    pKa2 :
        pKa2
    slope :
        slope
    E_infinity :
        E_infinity
    var :
        var
    """
    gaus = norm(0, var)
    pH_split = [[], [], []]
    E_split = [[], [], []]
    for _pH, _E in zip(pH, E):
        if _pH < pKa1:
            i = 0
        elif _pH < pKa2:
            i = 1
        else:
            i = 2
        pH_split[i].append(_pH)
        E_split[i].append(_E)

    p = 1.0
    for i in range(3):
        if len(pH_split[i]) > 0:
            x = np.array(pH_split[i])
            y_actual = np.array(E_split[i])
            assert x.shape[0] == y_actual.shape[0]
            y_pred = np.array(
                [value_at_ph(_pH, pKa1, pKa2, slope, E_infinity)
                 for _pH in pH_split[i]]
            )
            for j in range(x.shape[0]):
                p *= gaus.pdf(y_pred[j] - y_actual[j])

    return p


def postprocessing_pourbaix(
    data,
    save_path,
    var=0.01,
    M=30000,
    N=100,
):
    """generate_pourbaix_plot.

    Parameters
    ----------
    data :
        data
    save_path :
        save_path
    var :
        var
    M :
        M
    N :
        N
    """
    pH_test = [d["pH"] for d in data]
    E_test = [d["eV"] for d in data]
    print("DATA", data)
    print("ph_test", pH_test)
    print("E_test", E_test)

    pKa1_mean = np.array([])
    pKa1_stddev = np.array([])
    pKa2_mean = np.array([])
    pKa2_stddev = np.array([])

    P = np.zeros(M)
    E = np.zeros((M, N))
    pH = np.linspace(0, 14, N)
    pKa1 = np.zeros(M)
    pKa2 = np.zeros(M)
    slope = np.zeros(M)
    E_infinity = np.zeros(M)

    for i in range(M):
        pKa1[i] = np.random.uniform(pKa1_range[0], pKa1_range[1])
        pKa2[i] = np.random.uniform(max(pKa2_range[0], pKa1[i]), pKa2_range[1])
        slope[i] = np.random.uniform(slope_range[0], slope_range[1])
        E_infinity[i] = np.random.uniform(
            E_infinity_range[0], E_infinity_range[1])
        P[i] = gaus_prob(
            pH_test, E_test, pKa1[i], pKa2[i], slope[i], E_infinity[i], var=var
        )
        for j in range(N):
            E[i, j] = value_at_ph(pH[j], pKa1[i], pKa2[i],
                                  slope[i], E_infinity[i])

    X = np.dot(P.reshape(1, -1), E) / np.sum(P)
    STD_DEV = np.sqrt(
        np.dot(P.reshape(1, -1), (E - X.reshape(-1)) ** 2) / np.sum(P))

    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.plot(pH, X[0, :])
    ax1.fill_between(pH, X[0, :] - STD_DEV[0, :], X[0, :] + STD_DEV[0, :], alpha=0.3)
    ax1.scatter(pH_test, E_test)
    ax1.set_title('Pourbaix plot')
    ax1.set_xlabel('pH')
    ax1.set_ylabel('eV (mV)')


    pKa1_mean = np.append(pKa1_mean, np.sum(P * pKa1) / np.sum(P))
    pKa1_stddev = np.append(
        pKa1_stddev,
        np.sqrt(np.dot(P.reshape(1, -1),
                (pKa1 - pKa1_mean[-1]) ** 2) / np.sum(P)),
    )

    pKa2_mean = np.append(pKa2_mean, np.sum(P * pKa2) / np.sum(P))
    pKa2_stddev = np.append(
        pKa2_stddev,
        np.sqrt(np.dot(P.reshape(1, -1),
                (pKa2 - pKa2_mean[-1]) ** 2) / np.sum(P)),
    )

    slope_mean = np.sum(P * slope) / np.sum(P)
    slope_stddev = np.sqrt(np.sum(P * (slope - slope_mean) ** 2) / np.sum(P))


    ax2.plot(pH, norm.pdf(pH, pKa1_mean, pKa1_stddev))
    ax2.plot(pH, norm.pdf(pH, pKa2_mean, pKa2_stddev))
    ax2.set_title('pKa estimation')
    ax2.set_xlabel('pH')
    ax2.set_ylabel('Probability density function (PDF)')

    plt.tight_layout()
    print("saving to........", save_path)
    fig.savefig(save_path)
    plt.clf()
    plt.close()
    data = {
        "pKa1_mean": pKa1_mean,
        "pKa1_stddev": pKa1_stddev,
        "pKa2_mean": pKa2_mean,
        "pKa2_stddev": pKa2_stddev,
        "text": "The estimate for pKa1 is {:.3f} +/- {:.3f}. The estimate for pKa2 is {:.3f} +/- {:.3f}.".format(
            pKa1_mean[0], pKa1_stddev[0], pKa2_mean[0], pKa2_stddev[0]
        ),
    }
    return data
