import numpy as np
import scipy


def gauss(x: np.ndarray, amplitude: float, center: float, sigma: float):
    return amplitude * np.exp(-((x - center) ** 2) / (2 * sigma**2))


def scattered_gaussian_signal(
    time: np.ndarray,
    sigma: float,
    tau: float,
    amplitude: float = 1.0,
    peak_time: float = 0.0,
) -> np.ndarray:
    """The convolution between a gaussian distribution and a scattering function,
    $$
        \frac{e^{-t//\tau}}{\tau}U(t),
    $$
    where $U(t)$ is a step function which constrains the equation to time t > 0.
    The final equation is,
    $$
        I(t) = \frac{1}{2}\exp{-\frac{t}{\tau}
            + \frac{\sigma^2}{2\tau^2}}
                ( 1 + \text{erf} (
                    \frac{1}{\sqrt{2}\sigma} - \frac{\sigma}{\sqrt{2}\tau}
                    )
                )
    $$
    as described in [Oswald et. al. 2021](https://doi.org/10.1093/mnras/stab980).
    Args:
        time (np.ndarray): an array of observation time.
        sigma (float): the standard deviation of the data (the gaussian component)
        tau (float): the scattering time-scale (the scattering component)
        amplitude (float, optional): the amplitude of the signal. Defaults to 1.0.
        peak_time (float, optional): the peak of the signal. Defaults to 0.0.
    Returns:
        np.ndarray: y data
    """
    _time = time - peak_time
    first_term: np.ndarray = -(_time / tau) + (sigma**2) / (2 * (tau**2))
    second_term: np.ndarray = (_time / (np.sqrt(2) * sigma)) - (
        sigma / (np.sqrt(2) * tau)
    )
    return amplitude * 0.5 * np.exp(first_term) * (1 + scipy.special.erf(second_term))

