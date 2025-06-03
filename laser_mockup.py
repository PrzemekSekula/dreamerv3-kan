'''
This mockup simulates a laser device that generates an autocorrelation function (ACF) trace 

Consider a (gymnasium environment) -a laser device - where
- The input (action space) is a box of twenty values from 0 to 1023 (used to steer the device)
- The output are two lists of 1000 elements, describing sech^2-like pulse (ACF from laser)
Problem: we try to find the input which results in an ideal output

Mockup:

Parameters:
GENES = 20
GENE_MIN = 0
GENE_MAX = 1023
Action space (input)

N_SAMPLES = 1000
Observation Output

SCAN_RANGE_FS = 4000.0
Total range of time delay, in femtoseconds, covered by the ACF trace (from -2000 fs to +2000 fs).
Sets the width of the delay axis for the pulse. Affects the time window over which the pulse is analyzed and visualized.

FWHM_MIN_FS = 100.0, 
FWHM_MAX_FS = 400.0
Minimum and maximum full width at half maximum (FWHM) for the generated ACF pulse, in femtoseconds.
The best match (action == true_state) produces the narrowest ACF (FWHM_MIN_FS). 
As the action deviates from the optimum, the ACF broadens up to FWHM_MAX_FS. 
This models the degradation in pulse quality as the mask diverges from optimal.

PEDESTAL_MAX = 0.20
Maximum relative height (20%) of the "pedestal"—the constant background under the pulse.
When the action is far from optimal, a pedestal (flat background offset) is added to the ACF, up to 20% of the peak. 
The pedestal is proportional to the distance from the optimum.

NOISE_MAX_SD = 0.4
Maximum standard deviation of Gaussian noise added to the ACF.
At the optimum, the pulse is noise-free.

LAMBDA_MSE = 500.0
Weight factor for the MSE (mean squared error) term in the new fitness metric.
Controls the relative importance of the fit between the measured ACF and the ideal sech² 
shape in the new_fitness function. Larger values make the metric more sensitive to shape mismatches.
'''

import numpy as np
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import warnings


class MockupACFPulse:
    """Synthetic laser ACF generator with an internal secret optimum mask."""

    # ----- physical & numerical constants ----------------------------------
    GENES          = 20             # length of SLM mask
    GENE_MIN       = 0              # minimum value of each mask component
    GENE_MAX       = 1023           # maximum value of each mask component
    N_SAMPLES      = 1000           # points per trace (length of observation)
    SCAN_RANGE_FS  = 4000.0          # width of the delay axis for the pulse, window size
    FWHM_MIN_FS    = 100.0          # Minimum full width at half maximum (FWHM) for ACF puslse in femtoseconds
    FWHM_MAX_FS    = 400.0
    PEDESTAL_MAX   = 0.20            # Maximum relative height (20%) of the "pedestal"
    NOISE_MAX_SD   = 0.4           
    LAMBDA_MSE     = 0.5           # weight in the new metric (fs)

    # -----------------------------------------------------------------------

    def __init__(self, true_state: np.ndarray | None = None, seed: int | None = None):
        """Store the optimal mask and pre-compute the delay axis."""
        self.rng = np.random.default_rng(seed)
        if true_state is None:
            self.true_state = self.rng.integers(self.GENE_MIN,
                                                self.GENE_MAX + 1,
                                                size=self.GENES,
                                                endpoint=False)
        else:
            true_state = np.asarray(true_state, dtype=int)
            if true_state.shape != (self.GENES,):
                raise ValueError(f"true_state must have shape ({self.GENES},)")
            self.true_state = true_state
        # time (fs) axis centred at zero
        self.delay = np.linspace(-0.5*self.SCAN_RANGE_FS,
                                  0.5*self.SCAN_RANGE_FS,
                                  self.N_SAMPLES,
                                  dtype=float)
        self.img_shape=(64,64, 3)

    # ------------------------- low-level helpers ---------------------------

    @staticmethod
    def _sech_squared(x: np.ndarray, fwhm_fs: float) -> np.ndarray:
        """Return normalised sech² ACF with the requested FWHM."""
        k = 1.763 / fwhm_fs           # 1.763 ≃ 2 arccosh √2
        return 1.0 / np.cosh(k * x)**2

    def _distance(self, action: np.ndarray) -> float:
        """ℓ¹ distance between mask and optimum, normalised to [0,1]."""
        return np.sum(np.abs(action - self.true_state)) / (self.GENES * self.GENE_MAX)

    # --------------------------- public API --------------------------------

    def get_pulse(self, action: np.ndarray) -> tuple[list[float], list[float]]:
        """Generate the delay axis and synthetic ACF for a given mask."""
        action = np.asarray(action, dtype=int)
        if action.shape != (self.GENES,):
            raise ValueError(f"action must have shape ({self.GENES},)")
        if np.any((action < self.GENE_MIN) | (action > self.GENE_MAX)):
            raise ValueError("action components out of bounds")

        d = self._distance(action)                         # [0,1]

        # Width broadens with distance
        fwhm = self.FWHM_MIN_FS + (self.FWHM_MAX_FS - self.FWHM_MIN_FS) * d

        # Base sech² trace
        acf = self._sech_squared(self.delay, fwhm)

        # Pedestal and noise
        pedestal = self.PEDESTAL_MAX * d
        noise_sd = self.NOISE_MAX_SD * d
        acf = pedestal + (1.0 - pedestal) * acf
        acf += self.rng.normal(scale=noise_sd, size=acf.shape)
        acf = np.clip(acf, 0.0, 1.0)

        return self.delay.tolist(), acf.tolist()

    # ------------ metrics --------------------------------------------------

    def fitness(self, delay: np.ndarray, acf: np.ndarray) -> float:
        """Reproduce the original pulse_qual metric."""
        # normalise
        acf_norm = (acf - acf.min()) / (acf.max() - acf.min())
        peak_idx = int(np.argmax(acf_norm))

        # FWHM
        half = 0.5
        left_idx  = peak_idx - np.argmin(np.abs(acf_norm[peak_idx::-1] - half))
        right_idx = peak_idx + np.argmin(np.abs(acf_norm[peak_idx:]   - half))
        fwhm = (delay[right_idx] - delay[left_idx])        # fs (delay already in fs)

        # Areas
        area      = trapezoid(acf_norm, delay)
        fit       = self._sech_squared(delay, fwhm)
        area_fit  = trapezoid(fit, delay)

        return float(fwhm * (1.0 + abs(1.0 - area_fit/area))**2)

    def new_fitness_old(self, delay: np.ndarray, acf: np.ndarray) -> float:
        """
            Width plus MSE to the ideal sech² profile.
        """
        acf_norm = (acf - acf.min()) / (acf.max() - acf.min())
        peak_idx = int(np.argmax(acf_norm))

        # FWHM as above
        half = 0.5
        left_idx  = peak_idx - np.argmin(np.abs(acf_norm[peak_idx::-1] - half))
        right_idx = peak_idx + np.argmin(np.abs(acf_norm[peak_idx:]   - half))
        fwhm = (delay[right_idx] - delay[left_idx])

        ideal = self._sech_squared(delay, fwhm)
        mse   = np.mean((acf_norm - ideal)**2)

        return float(fwhm + self.LAMBDA_MSE * mse)

    def new_fitness(self, delay: np.ndarray, acf: np.ndarray) -> float:
        """
        Blend normalised FWHM and MSE (λ∈[0,1]) and map back to femtoseconds.
        Lower is better; perfect ≈ FWHM_MIN_FS, worst ≈ FWHM_MAX_FS.
        """
        # --------- trace normalisation ----------------------------------------
        acf_norm = (acf - acf.min()) / (acf.max() - acf.min())
        peak_idx = int(np.argmax(acf_norm))

        # --------- FWHM --------------------------------------------------------
        half = 0.5
        left_idx  = peak_idx - np.argmin(np.abs(acf_norm[peak_idx::-1] - half))
        right_idx = peak_idx + np.argmin(np.abs(acf_norm[peak_idx:]   - half))
        fwhm = delay[right_idx] - delay[left_idx]               # fs

        # FWHM → [0,1]
        width_range = self.FWHM_MAX_FS - self.FWHM_MIN_FS
        fwhm_norm = (fwhm - self.FWHM_MIN_FS) / width_range
        fwhm_norm = np.clip(fwhm_norm, 0.0, 1.0)

        # --------- shape error -------------------------------------------------
        ideal = self._sech_squared(delay, fwhm)
        mse = np.mean((acf_norm - ideal) ** 2)                  # already [0,1]

        # --------- weighted blend ---------------------------------------------
        lam = float(np.clip(self.LAMBDA_MSE, 0.0, 1.0))
        blend = (1.0 - lam) * fwhm_norm + lam * mse             # [0,1]

        # --------- back to femtoseconds ---------------------------------------
        score = self.FWHM_MIN_FS + blend * width_range
        return float(score)



    # ------------ rendering ------------------------------------------------

    def render(self, delay: np.ndarray, acf: np.ndarray) -> np.ndarray:
        """Return an RGB image (HxWx3) of the ACF plot, gym-style."""
        fig, ax = plt.subplots(figsize=(1, 1), dpi=64)
        ax.plot(delay, acf,lw=2)
        #ax.set_xlabel("Delay (fs)")
        #ax.set_ylabel("ACF (a.u.)")
        #ax.set_title("Mock-up ACF")
        ax.set_xlim(delay[0], delay[-1])
        ax.set_ylim(0, 1.05)
        ax.grid(True, lw=0.3)
        fig.canvas.draw()

        ax.set_axis_off()
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            # Set backgrounds transparent
            buf = BytesIO()
            plt.savefig(buf, format='png')
            plt.close(fig)
            buf.seek(0)


            # Load as PIL image and convert to RGB numpy array
            im = Image.open(buf).convert('RGB')
            im = im.resize(self.img_shape[:2][::-1], Image.BICUBIC)
            arr = np.array(im)
            assert arr.shape==self.img_shape, f"Expected shape {self.img_shape}, got {arr.shape}"
            return arr

if __name__ == "__main__":
    np.set_printoptions(threshold=20, edgeitems=3)

    # 1. Instantiate the mock device
    device = MockupACFPulse(seed=42)
    print("True mask:", device.true_state)

    # 2. Perfect action (should be the narrowest, cleanest pulse)
    delay_opt, acf_opt = device.get_pulse(device.true_state)
    f_old_opt = device.fitness(np.asarray(delay_opt), np.asarray(acf_opt))
    f_new_opt = device.new_fitness(np.asarray(delay_opt), np.asarray(acf_opt))
    print(f"Perfect mask → fitness_old={f_old_opt:.1f}, fitness_new={f_new_opt:.1f}")

    # 3. Random poor action
    random_action = np.random.randint(device.GENE_MIN,
                                      device.GENE_MAX + 1,
                                      size=device.GENES)
    delay_bad, acf_bad = device.get_pulse(random_action)
    f_old_bad = device.fitness(np.asarray(delay_bad), np.asarray(acf_bad))
    f_new_bad = device.new_fitness(np.asarray(delay_bad), np.asarray(acf_bad))
    print(f"Random mask  → fitness_old={f_old_bad:.1f}, fitness_new={f_new_bad:.1f}")

    # 4. Visualise both traces
    img_opt = device.render(np.asarray(delay_opt), np.asarray(acf_opt))
    img_bad = device.render(np.asarray(delay_bad), np.asarray(acf_bad))

    # If running inside gymnasium this image would be the observation.
    # Here we just show them with matplotlib for illustration:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 3))
    plt.imshow(img_opt)
    plt.axis("off")
    plt.title("Optimal mask trace")
    plt.show()

    plt.figure(figsize=(6, 3))
    plt.imshow(img_bad)
    plt.axis("off")
    plt.title("Random mask trace")
    plt.show()



