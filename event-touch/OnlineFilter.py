import numpy as np
import scipy.signal

class LowpassFilter(object):
    """
    DESCRIPTION
    """
    def __init__(self, input_length: int = 2, fpassband: float = 0.2, fstopband: float = 0.3, max_loss: float = 1, min_attenuation: float = 45):
        """
        Initialises lowpass filter with the input specifications using `scipy.signal.iirdesign()`.

        Parameters
        ----------
            `fpassband`         - passband edge frequency normalised w.r.t. the Nyquist frequency\n
            `fstopband`         - stopband edge frequency normalised w.r.t. the Nyquist frequency\n
            `max_loss`          - the maximum loss in the passband (dB)\n
            `min_attenuation`   - the minimum attenuation in the stopband (dB)\n
        """
        self.sos = scipy.signal.iirdesign(
            fpassband,
            fstopband,
            max_loss,
            min_attenuation,
            output='sos')

        self.n_sections = self.sos.shape[0]
        self.input_length = input_length
        # self.state = np.zeros((self.n_sections, 2))
        self.state = np.zeros((self.n_sections, 2, self.input_length))

    def filter(self, x):
        """
        Filter incoming data with cascaded second-order sections.
        """
        for s in range(self.n_sections):  # apply filter sections in sequence
            b0, b1, b2, a0, a1, a2 = self.sos[s, :]

            # compute difference equations of transposed direct form II
            y = b0*x + self.state[s, 0, :]
            self.state[s, 0, :] = b1*x - a1*y + self.state[s, 1, :]
            self.state[s, 1, :] = b2*x - a2*y
            x = y  # set biquad output as input of next filter section.

        return y