import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# Physical constants
R_d = 287.05       # J/(kg·K) dry air gas constant
R_v = 461.5        # J/(kg·K) water vapor gas constant
g = 9.80665        # m/s^2
L_DEFAULT = 0.0065  # K/m (standard tropospheric lapse rate)
eps = 0.622         # R_d / R_v


def saturation_vapor_pressure_pa(T_C):
    """Tetens approximate saturation vapor pressure over liquid water.

    T_C: temperature in degC (scalar or numpy array)
    returns: e_s in Pa
    """
    T_C = np.asarray(T_C)
    # Magnus parameters (valid roughly -40..+50 C)
    a = 17.27
    b = 237.3
    e_s_hPa = 6.1078 * np.exp(a * T_C / (b + T_C))
    return e_s_hPa * 100.0  # hPa -> Pa (6.112 was in hPa)


def air_density(T0_C, RH0, P0_pa, h0=0.0, h=0.0, lapse_rate=L_DEFAULT):
    """Compute air density at height h given conditions at height h0.

    Assumptions:
    - Temperature follows linear lapse: T(z) = T0 - lapse_rate*(z - h0)
    - Water vapor mixing ratio (mass of vapor per mass dry air) is conserved during vertical displacement
      (i.e., no condensation between h0 and h). This is a reasonable assumption for small vertical moves or
      unsaturated parcels. If condensation occurs, results will be approximate.

    Inputs:
    - T0_C: temperature at h0 in degC (scalar or numpy array for temperature sweeps)
    - RH0: relative humidity at h0 (0..1 or 0..100)
    - P0_pa: pressure at h0 in Pa
    - h0: reference altitude in meters
    - h: target altitude in meters (can be scalar or numpy array)
    - lapse_rate: K/m

    Returns air density rho (kg/m^3) at target altitude(s).
    """
    # normalize inputs to numpy arrays
    T0_C = np.asarray(T0_C)
    h = np.asarray(h)

    # normalize RH to fraction
    RH = float(RH0)
    if RH > 1.0 and RH <= 100.0:
        RH = RH / 100.0
    RH = float(RH)
    if not (0.0 <= RH <= 1.0):
        raise ValueError('RH must be between 0..1 or 0..100')

    # Temperature at target height (degC and K)
    T_h_C = T0_C - lapse_rate * (h - h0)
    T_h_K = T_h_C + 273.15

    # Pressure at target height using barometric formula for constant lapse rate
    # P(h) = P0 * (T(h)/T0)^(g/(R_d * lapse_rate))
    T0_K = T0_C + 273.15
    # avoid negative/zero temperatures in exponent
    ratio = T_h_K / T0_K
    # Allow broadcasting: if T0_C is array and h is array, numpy will handle shapes; otherwise rely on broadcasting rules.
    exponent = g / (R_d * lapse_rate)
    P_h = P0_pa * np.power(ratio, exponent)

    # compute initial saturation vapor pressure and actual vapor pressure at h0
    e_s0 = saturation_vapor_pressure_pa(T0_C)
    e0 = RH * e_s0

    # mixing ratio at h0 (kg_vapor / kg_dry_air)
    w0 = eps * e0 / (P0_pa - e0)

    # assume w conserved -> vapor partial pressure at h: e_h = w0 * P_h / (eps + w0)
    e_h = w0 * P_h / (eps + w0)

    # partial pressure of dry air
    Pd_h = P_h - e_h

    # densities
    rho_dry = Pd_h / (R_d * T_h_K)
    rho_vapor = e_h / (R_v * T_h_K)

    rho = rho_dry + rho_vapor
    return rho


def plot_density_vs_temperature(
    temps=np.linspace(-10, 40, 501),
    humidities=(0, 20, 40, 60, 80, 100),
    pressure=101325.0,
    h0=0.0,
    h=None,
    out_path=None,
):
    """Plot air density vs temperature for a set of relative humidities.

    - temps: array of temperatures (degC) at reference height h0
    - humidities: iterable of RH values (percent or 0..1)
    - pressure: pressure at h0 in Pa
    - h0: reference altitude for temps and pressure
    - h: target altitude at which density is computed. If None, use h0 (i.e., density at same altitude)
    """
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(8, 5))

    if h is None:
        h = h0

    for Rh in humidities:
        rho = air_density(temps, Rh, pressure, h0=h0, h=h)
        ax.plot(temps, rho, label=f'RH={Rh}%')

    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Air density (kg/m³)')
    title_h = f' at h={h} m' if np.any(np.asarray(h) != h0) else ''
    ax.set_title(f'Air density vs Temperature (P={pressure:.0f} Pa){title_h}')
    ax.legend(title='Relative Humidity')

    if out_path is None:
        out_path = Path(__file__).with_name('density_vs_temperature.png')

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    return fig, ax


def plot_density_vs_height(
    heights=np.linspace(0, 5000, 501),
    T0_C=15.0,
    humidities=(0, 20, 40, 60, 80, 100),
    pressure=101325.0,
    h0=0.0,
    out_path=None,
):
    """Plot air density vs height for several relative humidities.

    - heights: array of target heights (m)
    - T0_C: temperature at reference height h0 (degC)
    - humidities: iterable of RH values (percent or 0..1)
    - pressure: pressure at h0 in Pa
    - h0: reference altitude (m)
    """
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(8, 5))

    for Rh in humidities:
        rho = air_density(T0_C, Rh, pressure, h0=h0, h=heights)
        ax.plot(heights, rho, label=f'RH={Rh}%')

    ax.set_xlabel('Height (m)')
    ax.set_ylabel('Air density (kg/m³)')
    ax.set_title(f'Air density vs Height (T@{h0}={T0_C} °C, P@{h0}={pressure:.0f} Pa)')
    ax.legend(title='Relative Humidity')

    if out_path is None:
        out_path = Path(__file__).with_name('density_vs_height.png')

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    return fig, ax


def main():
    # Example usage: create both plots and save to repository folder
    temps = np.linspace(-20, 50, 701)
    humidities = [0, 20, 40, 60, 80, 100]
    pressure = 101325.0

    fig1, ax1 = plot_density_vs_temperature(temps=temps, humidities=humidities, pressure=pressure, h0=0.0, h=0.0)
    print('Saved:', Path(__file__).with_name('density_vs_temperature.png'))

    heights = np.linspace(0, 5000, 501)
    fig2, ax2 = plot_density_vs_height(heights=heights, T0_C=15.0, humidities=humidities, pressure=pressure, h0=0.0)
    print('Saved:', Path(__file__).with_name('density_vs_height.png'))

    try:
        plt.show()
    except Exception:
        pass


if __name__ == '__main__':
    main()
