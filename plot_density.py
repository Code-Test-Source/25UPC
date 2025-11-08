import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import the provided function for reference (we will use a corrected, vectorized wrapper below)
try:
    from air import air_density as air_density_original
except Exception:
    air_density_original = None


def air_density_corrected(T, Rh, P=101325.0):
    """
    Vectorized/corrected air density calculation based on the logic in `air.air_density`.

    Assumptions/notes:
    - T: temperature in degrees Celsius (scalar or numpy array)
    - Rh: relative humidity in percent (0-100)
    - P: absolute atmospheric pressure in Pascals (Pa). Default 101325 Pa.

    The original `air_density` used Tetens formula but returned saturation vapor
    pressure in kPa. Here we explicitly convert to Pa to keep units consistent.
    """
    R = 287.05   # J/(kg·K) dry air
    Rv = 461.5   # J/(kg·K) water vapor

    T = np.asarray(T)
    T_k = T + 273.15

    # Tetens formula (gives kPa if using 0.61078) -> convert to Pa by *1000
    Es_kpa = 0.61078 * np.exp((17.27 * T / (T + 237.3)))
    Es = Es_kpa * 1000.0  # Pa

    E = (Rh / 100.0) * Es
    Pd = P - E

    rho_dry = Pd / (R * T_k)
    rho_vapor = E / (Rv * T_k)

    return rho_dry + rho_vapor


def plot_density_vs_temperature(
    temps=np.linspace(-10, 40, 501),
    humidities=(0, 20, 40, 60, 80, 100),
    pressure=101325.0,
    out_path=None,
):
    """Plot density vs temperature for several relative humidities at fixed pressure.

    Returns the figure and axes objects.
    """
    # Use a built-in matplotlib style to avoid optional seaborn dependency
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(8, 5))

    for Rh in humidities:
        rho = air_density_corrected(temps, Rh, pressure)
        ax.plot(temps, rho, label=f'RH={Rh}%')

    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Air density (kg/m³)')
    ax.set_title(f'Air density vs Temperature (P={pressure:.0f} Pa)')
    ax.legend(title='Relative Humidity')

    if out_path is None:
        out_path = Path(__file__).with_name('density_vs_temperature.png')

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)

    return fig, ax


def main():
    temps = np.linspace(-20, 50, 701)
    humidities = [0, 20, 40, 60, 80, 100]
    pressure = 101325.0

    fig, ax = plot_density_vs_temperature(temps=temps, humidities=humidities, pressure=pressure)
    print('Plot saved to:', Path(__file__).with_name('density_vs_temperature.png'))
    try:
        plt.show()
    except Exception:
        # In headless environments this may fail; it's okay because the file is saved.
        pass


if __name__ == '__main__':
    main()
