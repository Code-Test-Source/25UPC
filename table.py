import runpy
import argparse
import csv
import math
import time
import os

# Load try.py namespace (ArtillerySolver3D defined there)
ns = runpy.run_path(os.path.join(os.path.dirname(__file__), 'try.py'))
ArtillerySolver3D = ns.get('ArtillerySolver3D')
if ArtillerySolver3D is None:
    raise ImportError('ArtillerySolver3D not found in try.py')


def polar_wind(mag, deg):
    rad = math.radians(deg)
    return [mag * math.cos(rad), mag * math.sin(rad), 0.0]


def quick_grid_solve(solver, distance, initial_altitude, wind_vec, v_min=100, v_max=450, v_step=10, elev_step=2):
    """
    Simple grid search over (v0, elevation, azimuth) to find a reasonably-good firing solution.
    This deliberately avoids expensive optimizers and uses coarse sampling to be fast and predictable.

    Returns (v0, elevation_deg, azimuth_deg, error_m)
    """
    best = (None, None, None, float('inf'))

    # Azimuth choices: if wind negligible, force 0; otherwise allow small compensations
    wind_mag = math.hypot(wind_vec[0], wind_vec[1])
    if wind_mag < 0.5:
        az_choices = [0.0]
    else:
        az_choices = list(range(-8, 9, 2))  # -8,-6,...,0,...,8 deg

    v_vals = list(range(int(v_min), int(min(v_max, solver.max_muzzle_velocity)) + 1, int(v_step)))
    elev_vals = list(range(1, 81, max(1, int(elev_step))))

    # Fast coarse evaluation loop
    for v0 in v_vals:
        for elev in elev_vals:
            for az in az_choices:
                sol = solver.simulate_trajectory(v0, elev, az, wind_vec, initial_altitude, distance, max_time=60)
                ipos, itime, err = solver.calculate_impact(sol, distance)
                if ipos is None:
                    continue
                # Accept if better error, prefer lower v0 on tie
                if err < best[3] - 1e-6 or (abs(err - best[3]) <= 1e-6 and (best[0] is None or v0 < best[0])):
                    best = (v0, elev, az, err)

    return best


def build_table(output_csv, distances, theta_vals, v_vals, initial_altitude=500, max_cases=None):
    # Build task list
    tasks = []
    for dist in distances:
        for theta in theta_vals:
            for v in v_vals:
                tasks.append((dist, theta, v, initial_altitude))
    total = len(tasks)

    fieldnames = ['distance_m', 'wind_v_m_s', 'wind_theta_deg', 'muzzle_velocity_m_s', 'elevation_deg', 'azimuth_deg', 'error_m']
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)

        print(f"Running {total} cases (distances={len(distances)}, thetas={len(theta_vals)}, winds={len(v_vals)})")
        start_time = time.time()

        # Use multiprocessing to parallelize; worker initializer will create a solver instance per process
        def init_worker():
            global WORKER_SOLVER_CLASS
            global WORKER_SOLVER
            ns = runpy.run_path(os.path.join(os.path.dirname(__file__), 'try.py'))
            WORKER_SOLVER_CLASS = ns.get('ArtillerySolver3D')
            if WORKER_SOLVER_CLASS is None:
                raise ImportError('ArtillerySolver3D not found in try.py (worker)')
            WORKER_SOLVER = WORKER_SOLVER_CLASS()

        def worker(task):
            dist, theta, v, alt = task
            wind_vec = polar_wind(v, theta)
            try:
                # call quick_grid_solve using the per-process solver instance
                best = quick_grid_solve(WORKER_SOLVER, dist, alt, wind_vec,
                                        v_min=100, v_max=WORKER_SOLVER.max_muzzle_velocity,
                                        v_step=10, elev_step=2)
                if best is None:
                    return (dist, v, theta, '', '', '', float('nan'))
                v0, elev, az, err = best
                return (dist, v, theta, v0, elev, az, err)
            except Exception:
                return (dist, v, theta, '', '', '', float('nan'))

        # Number of processes: default to CPU count
        nprocs = max(1, multiprocessing.cpu_count() - 1)
        with multiprocessing.Pool(processes=nprocs, initializer=init_worker) as pool:
            for i, res in enumerate(pool.imap_unordered(worker, tasks), start=1):
                writer.writerow(res)
                # progress every 50 cases
                if i % 50 == 0 or i == total:
                    elapsed = time.time() - start_time
                    eta = (elapsed / i) * (total - i) if i > 0 else 0
                    print(f"  {i}/{total} done — elapsed {elapsed:.1f}s, ETA {eta:.1f}s")

        elapsed = time.time() - start_time
        print(f"Completed {total} cases in {elapsed:.1f}s. Output written to {output_csv}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build firing table using ArtillerySolver3D from try.py')
    parser.add_argument('--dmin', type=int, default=1000, help='min distance (m)')
    parser.add_argument('--dmax', type=int, default=1500, help='max distance (m)')
    parser.add_argument('--dstep', type=int, default=100, help='distance step (m)')
    parser.add_argument('--thetastep', type=int, default=30, help='wind direction step (deg)')
    parser.add_argument('--vstep', type=int, default=5, help='wind speed step (m/s)')
    parser.add_argument('--vmax', type=int, default=30, help='max wind speed (m/s)')
    parser.add_argument('--alt', type=int, default=500, help='initial altitude (m)')
    parser.add_argument('--out', type=str, default='firing_table.csv', help='output CSV file')
    parser.add_argument('--maxcases', type=int, default=None, help='stop after this many cases (for testing)')
    args = parser.parse_args()

    distances = list(range(args.dmin, args.dmax + 1, args.dstep))
    theta_vals = list(range(0, 360, args.thetastep))
    v_vals = list(range(0, args.vmax + 1, args.vstep))

    build_table(args.out, distances, theta_vals, v_vals, initial_altitude=args.alt, max_cases=args.maxcases)
