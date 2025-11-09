import csv
import os
import math
import time
import numpy as np
from math import isfinite
from fastest_angle import ArtillerySolver3D
from scipy.optimize import minimize
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from scipy.spatial import qhull


def _safe_griddata(pts_arr, vals_arr, query_arr):
    """Try linear griddata, fall back to nearest or manual nearest if Qhull fails."""
    try:
        out = griddata(pts_arr, vals_arr, query_arr, method='linear')
        if out is None or np.isnan(out).any():
            out = griddata(pts_arr, vals_arr, query_arr, method='nearest')
        return out
    except Exception:
        # Qhull or other failure -> manual nearest neighbor
        try:
            dif = pts_arr - query_arr.reshape(1, -1)
            dists = np.sqrt(np.sum(dif**2, axis=1))
            idx = int(np.argmin(dists))
            return np.array([vals_arr[idx]])
        except Exception:
            return None


def expand_table_simulation(ranges, wind_speeds, wind_dirs, output_csv='firing_table_expanded.csv',
                            initial_altitude=500, coarse_v_step=10, coarse_elev_step=2,
                            air_density=1.2):
    """
    Delicate physics-based expansion: small coarse grid around a vacuum seed,
    writes each cell immediately and prints concise per-cell result.
    """
    solver = ArtillerySolver3D()
    # set air density for this run and update derived drag constant
    solver.air_density = float(air_density)
    solver.k_drag = 0.5 * solver.air_density * solver.drag_coeff * solver.area

    header = ['distance_m', 'wind_speed_m_s', 'wind_dir_deg', 'muzzle_velocity_m_s',
              'elevation_deg', 'azimuth_deg', 'error_m', 'flight_time_s']

    rows = []

    # vacuum seed parameters (use elevation cap 25 deg)
    theta_deg_seed = 25.0
    g = solver.gravity
    theta_rad = math.radians(theta_deg_seed)
    sin_2theta = math.sin(2.0 * theta_rad)

    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for d in ranges:
            # compute a fast vacuum-based velocity seed
            if sin_2theta <= 1e-8:
                v_seed = float(solver.max_muzzle_velocity)
            else:
                v_seed = math.sqrt(max(0.0, d * g / sin_2theta))
                v_seed = min(max(100.0, v_seed), float(solver.max_muzzle_velocity))

            for ws in wind_speeds:
                for wd in wind_dirs:
                    rad = math.radians(wd)
                    wind_v = np.array([ws * math.cos(rad), ws * math.sin(rad), 0.0])

                    # search window
                    v_min = max(100.0, v_seed - 60.0)
                    v_max = min(float(solver.max_muzzle_velocity), v_seed + 60.0)
                    v_vals = np.arange(v_min, v_max + 1e-6, coarse_v_step)
                    elev_min = max(1.0, theta_deg_seed - 10.0)
                    elev_max = 25.0
                    elev_vals = np.arange(elev_min, elev_max + 1e-6, coarse_elev_step)

                    az_choices = [0.0, -5.0, 5.0, -10.0, 10.0]

                    best = None
                    best_time = None
                    best_err = float('inf')

                    for v in v_vals:
                        for elev in elev_vals:
                            for az in az_choices:
                                sol = solver.simulate_trajectory(v, elev, az, wind_v, initial_altitude, d)
                                impact, t_imp, err = solver.calculate_impact(sol, d)
                                if impact is not None and isfinite(err) and err < best_err:
                                    best_err = err
                                    best = (v, elev, az)
                                    best_time = t_imp

                    if best is not None:
                        chosen = best
                        chosen_time = float(best_time) if best_time is not None else float('nan')
                        chosen_err = float(best_err)
                    else:
                        chosen = (v_seed, theta_deg_seed, 0.0)
                        sol = solver.simulate_trajectory(chosen[0], chosen[1], chosen[2], wind_v, initial_altitude, d)
                        impact, t_imp, err = solver.calculate_impact(sol, d)
                        chosen_time = float(t_imp) if t_imp is not None else float('nan')
                        chosen_err = float(err) if isfinite(err) else float('nan')

                    writer.writerow([
                        int(d), float(ws), float(wd), float(chosen[0]), float(chosen[1]), float(chosen[2]),
                        chosen_err, chosen_time
                    ])
                    f.flush()

                    print(f"RESULT -> Range {d}m | Wind {ws}m/s @{wd}° | V={chosen[0]:.1f} m/s | Elev={chosen[1]:.1f}° | Az={chosen[2]:.1f}° | Err={chosen_err:.2f} m | T={chosen_time:.2f}s")

                    rows.append((int(d), float(ws), float(wd), float(chosen[0]), float(chosen[1]), float(chosen[2]), chosen_err, chosen_time))

    # final summary
    print('\nExpanded table (simulation) written to', output_csv)
    print('Summary per range:')
    for d in ranges:
        subset = [r for r in rows if r[0] == d]
        if not subset:
            continue
        avg_v = np.mean([r[3] for r in subset])
        avg_e = np.mean([r[4] for r in subset])
        avg_err = np.mean([r[6] for r in subset])
        print(f"{d:8d} | {avg_v:17.1f} | {avg_e:16.1f} | {avg_err:12.1f}")


def expand_from_existing_csv(existing_csv, ranges, wind_speeds, wind_dirs, output_csv='firing_table_interpolated.csv'):
    """
    Fast expansion by interpolating parameters from an existing table.
    Existing CSV must have header with columns: distance_m, wind_speed_m_s, wind_dir_deg,
    muzzle_velocity_m_s, elevation_deg, azimuth_deg (case-insensitive fallback names supported).
    """
    if not os.path.exists(existing_csv):
        return False

    pts = []
    vel = []
    elev = []
    az_vx = []
    az_vy = []
    err = []

    with open(existing_csv, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                d = float(row.get('distance_m') or row.get('distance') or row.get('Range') )
                ws = float(row.get('wind_speed_m_s') or row.get('wind_magnitude') or row.get('wind'))
                wd = float(row.get('wind_dir_deg') or row.get('wind_direction') or row.get('wind_dir') or 0.0)
                v = float(row.get('muzzle_velocity_m_s') or row.get('muzzle_velocity') or row.get('Velocity') or 0.0)
                e = float(row.get('elevation_deg') or row.get('elevation') or row.get('Elevation') or 0.0)
                az = float(row.get('azimuth_deg') or row.get('azimuth') or row.get('Azimuth') or 0.0)
                er = float(row.get('error_m') or row.get('error') or row.get('Error') or 0.0)
            except Exception:
                continue

            rad = math.radians(wd)
            wx = ws * math.cos(rad)
            wy = ws * math.sin(rad)

            az_rad = math.radians(az)
            ax = math.cos(az_rad)
            ay = math.sin(az_rad)

            pts.append((d, wx, wy))
            vel.append(v)
            elev.append(e)
            az_vx.append(ax)
            az_vy.append(ay)
            err.append(er)

    if len(pts) < 4:
        return False

    pts = np.array(pts)
    vel = np.array(vel)
    elev = np.array(elev)
    az_vx = np.array(az_vx)
    az_vy = np.array(az_vy)
    err = np.array(err)

    rows_out = []
    with open(output_csv, 'w', newline='') as outf:
        writer = csv.writer(outf)
        writer.writerow(['distance_m', 'wind_speed_m_s', 'wind_dir_deg', 'muzzle_velocity_m_s', 'elevation_deg', 'azimuth_deg', 'error_m'])

        for d in ranges:
            for ws in wind_speeds:
                for wd in wind_dirs:
                    rad = math.radians(wd)
                    wx = ws * math.cos(rad)
                    wy = ws * math.sin(rad)
                    query = np.array([[d, wx, wy]])

                    v_q = _safe_griddata(pts, vel, query)
                    e_q = _safe_griddata(pts, elev, query)
                    ax_q = _safe_griddata(pts, az_vx, query)
                    ay_q = _safe_griddata(pts, az_vy, query)
                    err_q = _safe_griddata(pts, err, query)

                    # final fallback to nearest if any are still None
                    if v_q is None or np.isnan(v_q).any():
                        v_q = _safe_griddata(pts, vel, query)
                    if e_q is None or np.isnan(e_q).any():
                        e_q = _safe_griddata(pts, elev, query)
                    if ax_q is None or np.isnan(ax_q).any():
                        ax_q = _safe_griddata(pts, az_vx, query)
                    if ay_q is None or np.isnan(ay_q).any():
                        ay_q = _safe_griddata(pts, az_vy, query)
                    if err_q is None or np.isnan(err_q).any():
                        err_q = _safe_griddata(pts, err, query)

                    vx = float(v_q.flatten()[0])
                    el = float(e_q.flatten()[0])
                    axv = float(ax_q.flatten()[0])
                    ayv = float(ay_q.flatten()[0])
                    errv = float(err_q.flatten()[0])

                    az_deg = math.degrees(math.atan2(ayv, axv))

                    writer.writerow([int(d), float(ws), float(wd), vx, el, az_deg, errv])
                    outf.flush()
                    print(f"INTERP -> Range {d}m | Wind {ws}m/s @{wd}° | V={vx:.1f} m/s | Elev={el:.1f}° | Az={az_deg:.1f}° | Err={errv:.2f} m")
                    rows_out.append((d, ws, wd, vx, el, az_deg, errv))

    print('Interpolated table written to', output_csv)
    return True


def expand_over_densities(densities, ranges, wind_speeds, wind_dirs, out_prefix='firing_table_expanded_rho',
                          initial_altitude=500, coarse_v_step=10, coarse_elev_step=2):
    """
    Run delicate expansion for multiple air densities and plot summary.
    Writes one CSV per density and a PNG summary plot showing avg muzzle velocity
    vs range for each density.
    """
    summary = {}
    for rho in densities:
        out_csv = f"{out_prefix}{rho:.2f}.csv"
        print(f"\nStarting expansion for air density={rho:.2f} kg/m^3 -> output: {out_csv}")
        expand_table_simulation(ranges, wind_speeds, wind_dirs, output_csv=out_csv,
                                initial_altitude=initial_altitude, coarse_v_step=coarse_v_step,
                                coarse_elev_step=coarse_elev_step, air_density=rho)

        # read back CSV to compute avg per range
        data = {}
        try:
            with open(out_csv, newline='') as f:
                rdr = csv.DictReader(f)
                for row in rdr:
                    try:
                        d = int(float(row.get('distance_m') or row.get('distance') or row.get('Range')))
                        v = float(row.get('muzzle_velocity_m_s') or row.get('muzzle_velocity') or row.get('Velocity'))
                    except Exception:
                        continue
                    data.setdefault(d, []).append(v)
        except Exception as e:
            print('Could not read back', out_csv, e)
            continue

        avg_per_range = {d: float(np.mean(vs)) for d, vs in data.items()}
        summary[rho] = avg_per_range

    # Plot summary: avg velocity vs range, one line per density
    if summary:
        plt.figure(figsize=(10, 6))
        for rho, averages in sorted(summary.items()):
            xs = sorted(averages.keys())
            ys = [averages[x] for x in xs]
            plt.plot(xs, ys, marker='o', label=f'rho={rho:.2f}')
        plt.xlabel('Range (m)')
        plt.ylabel('Avg Muzzle Velocity (m/s)')
        plt.title('Avg Muzzle Velocity vs Range for different air densities')
        plt.grid(True, alpha=0.3)
        plt.legend()
        out_png = out_prefix + '_summary.png'
        plt.tight_layout()
        plt.savefig(out_png)
        print('\nSaved density-summary plot to', out_png)


if __name__ == '__main__':
    ranges = [1000, 1100, 1200, 1300, 1400, 1500]
    wind_speeds = list(range(0, 11))  # 0 to 10 m/s
    wind_dirs = list(range(0, 360, 10))  # every 10 degrees

    # densities to evaluate (0.8 to 1.4 kg/m^3)
    densities = list(np.linspace(0.8, 1.4, 7))

    # If an existing expanded table exists, try fast interpolation first
    existing = 'firing_table_expanded.csv'
    used = expand_from_existing_csv(existing, ranges, wind_speeds, wind_dirs)
    if not used:
        # run delicate simulation across densities (this will produce one CSV per density)
        expand_over_densities(densities, ranges, wind_speeds, wind_dirs)
