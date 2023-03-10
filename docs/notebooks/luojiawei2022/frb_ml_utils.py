import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from scipy.integrate import quad
from scipy.optimize import fsolve
from sklearn.metrics import fbeta_score, make_scorer
from matplotlib import pyplot as plt

H_0 = 67.4*1000*100  # cm s^-1 Mpc^-1
c = 29979245800 # cm s^-1
Mpc_to_cm = 3.085677581491367E24
Gyr_to_s = 3.15576E16
Hubble_time = 1/(H_0 / Mpc_to_cm * Gyr_to_s)  # Gyr
Hubble_distance = c/H_0  # Mpc
Omega_b = 0.0224/((H_0)/1000/100/100)**2
Omega_m = 0.315
Omega_Lambda = 0.685
f_IGM = 0.83
chi = 7/8
G = 6.6743e-8 # cm^3 g^-1 s^-2
m_p = 1.67262192e-24 # g
dm_factor = 3*c*H_0/(Mpc_to_cm)**2*1e6*Omega_b*f_IGM*chi/(8*np.pi*G*m_p)
DM_host_lab = 70.0 # pc cm^-3
DM_halo = 30.0

f2_score = make_scorer(fbeta_score, beta=0.5)


def comoving_distance_at_z(z): # Mpc
    zp1 = 1.0+z
    h0_up = np.sqrt(1+Omega_m/Omega_Lambda) * scipy.special.hyp2f1(1/3,1/2,4/3,-Omega_m/Omega_Lambda)
    hz_up = zp1 * np.sqrt(1+Omega_m*zp1**3/Omega_Lambda) * scipy.special.hyp2f1(1/3,1/2,4/3,-Omega_m*zp1**3/Omega_Lambda)
    h0_down = np.sqrt(Omega_Lambda + Omega_m)
    hz_down = np.sqrt(Omega_Lambda + Omega_m * zp1**3)
    return Hubble_distance * (hz_up/hz_down-h0_up/h0_down)


def luminosity_distance_at_z(z): # Mpc
    return (1. + z) * comoving_distance_at_z(z)


def angular_diameter_distance_at_z(z): # Mpc
    return comoving_distance_at_z(z) / (1. + z)


def DM_int(z):
    return (1+z)/(np.sqrt(Omega_m*(1+z)**3+Omega_Lambda))


def DM_from_z(z): # pc cm^-3
    return dm_factor * quad(DM_int,0,z)[0]


def excess_DM(z):
    return DM_halo + DM_from_z(z) + DM_host_lab/(1+z)


def z_from_DM(DM):
    return fsolve(lambda z:excess_DM(z)-DM,0)[0]


def load_chime():
    CHIME = pd.read_csv('chimefrbcat1.csv')

    CHIME.drop([364,365,401,402,512,513],inplace=True) # No flux measurement
    CHIME.reset_index(drop=True,inplace=True)
    
    z_values = np.zeros(len(CHIME))
    for i,frb in CHIME.iterrows():
        DM = frb['dm_exc_ne2001']
        z = max(0.00225301,z_from_DM(DM)) # z=0.00225301, d_L=10 Mpc
        z_values[i] = z
        if z<=0.00225301:
            print(i,DM,z,frb['tns_name'])

    z_values[CHIME['repeater_name']=='FRB20121102A'] = 0.19273 # FRB 20121102A
    z_values[CHIME['repeater_name']=='FRB20180916B'] = 0.0337 # FRB 20180916B

    brightness_temperature = 1.1e35 * CHIME['flux'] * (CHIME['bc_width']*1000)**(-2) * (CHIME['peak_freq']/1000)**(-2) * (luminosity_distance_at_z(z_values)/1000)**2 / (1+z_values)
    brightness_temperature = brightness_temperature.to_numpy()
    rest_width = CHIME['width_fitb']/(1+z_values)
    frequency_bandwidth = (CHIME['high_freq'] - CHIME['low_freq']) * (1+z_values)
    energy = 1e-23 * CHIME['peak_freq'] * 1e6 * CHIME['fluence'] / 1000 * (4*np.pi*(luminosity_distance_at_z(z_values)*Mpc_to_cm)**2) / (1+z_values)

    CHIME['redshift'] = z_values
    CHIME['bright_temp'] = brightness_temperature
    CHIME['rest_width'] = rest_width
    CHIME['freq_width'] = frequency_bandwidth
    CHIME['energy'] = energy
    CHIME['repeating'] = ['repeating' if row['repeater_name'] != '-9999' else 'non-repeating' for i,row in CHIME.iterrows()]

    return CHIME


def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):

    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names) == cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in (cf / np.c_[np.sum(cf,axis=1),np.sum(cf,axis=1)]).flatten()]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        # Accuracy is sum of diagonal divided by total observations
        accuracy = np.trace(cf) / float(np.sum(cf))

        # if it is a binary confusion matrix, show some more stats
        if len(cf) == 2:
            # Metrics for Binary Confusion Matrices
            precision = cf[1, 1] / sum(cf[:, 1])
            recall = cf[1, 1] / sum(cf[1, :])
            f1_score = 2 * precision * recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy, precision, recall, f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize == None:
        # Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks == False:
        # Do not show categories if xyticks is False
        categories = False

    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    res = sns.heatmap(cf, annot=box_labels, fmt="", cmap=cmap, cbar=cbar, xticklabels=categories, yticklabels=categories)
    for _, spine in res.spines.items():
        spine.set_visible(True)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)

    if title:
        plt.title(title)
