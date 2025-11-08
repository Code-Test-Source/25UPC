import numpy as np
import matplotlib.pyplot as plt
import json
import csv
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, differential_evolution, Bounds

class ArtillerySolver3D:
    def __init__(self):
        # Physical constants
        self.mass = 5.0                    # Projectile mass [kg]
        self.diameter = 0.11               # Projectile diameter [m]
        self.radius = self.diameter / 2    # Projectile radius [m]
        self.area = np.pi * self.radius**2 # Cross-sectional area [m²]
        self.drag_coeff = 0.47            # Sphere drag coefficient
        self.air_density = 1.2             # Air density [kg/m³]
        self.gravity = 9.807                 # Gravity [m/s²]
        # Speed of sound used to compute Mach number for high-speed drag model [m/s]
        # Use 340 m/s as a practical threshold (close to standard sea-level value)
        self.speed_of_sound = 340.0
        self.k_drag = 0.5 * self.air_density * self.drag_coeff * self.area
        
        # Cannon constraints
        self.max_muzzle_velocity = 450     # Maximum muzzle velocity [m/s]
        
        # Create a robust ground-collision event (bound function with attributes)
        def _ground_collision_event(t, state, *args):
            return state[2]

        _ground_collision_event.terminal = True
        _ground_collision_event.direction = -1
        self.ground_collision_event = _ground_collision_event
    def wind_force(self, velocity, wind_velocity):
                """Calculate wind-induced force on projectile.

                Below ~340 m/s we use the empirical speed-dependent Cd(s). Above that
                threshold we switch to a Mach-based Cd(Ma) model.
                """
                relative_velocity = velocity - wind_velocity
                speed = np.linalg.norm(relative_velocity)

                if speed < 1e-10:
                        return np.zeros(3)

                # Select drag-coefficient model depending on speed
                if speed <= 340.0:
                        cd = self.drag_coeff_for_speed(speed)
                else:
                        mach = speed / float(self.speed_of_sound)
                        cd = self.drag_coeff_for_mach(mach)

                # Drag force magnitude (quadratic in speed)
                drag_magnitude = 0.5 * self.air_density * cd * self.area * speed**2

                # Direction of drag force (opposite to relative velocity)
                drag_direction = -relative_velocity / speed

                return drag_magnitude * drag_direction

    def drag_coeff_for_mach(self, mach):
                """Estimate drag coefficient Cd as a function of Mach number.

                This is a simple piecewise-linear approximation intended to capture the
                transonic/supersonic rise in drag coefficient near Ma~1 and then a
                gradual decline at higher Mach numbers. Anchors are tunable.

                NOTES/ASSUMPTIONS:
                - We use a pragmatic anchor set because a full compressible-flow Cd(Ma)
                    curve requires experimental data and depends on projectile shape.
                - Anchors (Ma -> Cd): 0.95->0.50, 1.00->1.20 (transonic spike),
                    1.20->0.90, 2.00->0.50, 5.00->0.30
                - If you have measured Cd vs Ma for your projectile, replace these
                    anchor points with that dataset.
                """
                m = max(0.0, float(mach))
                # piecewise linear interpolation over Mach anchors
                return float(np.interp(m, [0.95, 1.00, 1.20, 2.00, 5.00], [0.50, 1.20, 0.90, 0.50, 0.30]))

    def drag_coeff_for_speed(self, speed):
        """Return an estimated drag coefficient Cd as a function of relative speed.

        This uses a simple piecewise-linear interpolation between three anchor points
        to approximate the effect of changing flow/turbulence regimes.
        Tunable anchor points: (0 m/s -> 0.9), (150 m/s -> 0.47), (500 m/s -> 0.1).
        """
        # clamp speed to non-negative
        s = max(0.0, float(speed))
        # piecewise linear interpolation
        return float(np.interp(s, [0.0, 150.0, 500.0], [0.9, 0.47, 0.1]))
    
    def projectile_dynamics(self, t, state, muzzle_velocity, elevation_angle, azimuth_angle, wind_velocity):
        """
        3D Projectile motion equations with wind effects
        state = [x, y, z, vx, vy, vz]
        """
        position = state[:3]
        velocity = state[3:]
        
        # Convert angles to radians
        elevation_rad = np.radians(elevation_angle)
        azimuth_rad = np.radians(azimuth_angle)
        
        # Initialize acceleration
        acceleration = np.zeros(3)
        
        # Gravity (only in z-direction)
        acceleration[2] = -self.gravity
        
        # Aerodynamic drag (always apply; wind_velocity is the air speed vector)
        wind_force_vec = self.wind_force(velocity, wind_velocity)
        acceleration += wind_force_vec / self.mass
        
        # Combine derivatives
        derivatives = np.concatenate([velocity, acceleration])
        
        return derivatives
    
    def ground_collision_event(self, t, state, *args):
        """Event for when projectile hits the ground"""
        return state[2]  # Returns 0 when z=0 (ground level)
    
    ground_collision_event.terminal = True
    ground_collision_event.direction = -1
    
    def simulate_trajectory(self, muzzle_velocity, elevation_angle, azimuth_angle, 
                          wind_velocity, initial_altitude, target_distance, max_time=60):
        """
        Simulate projectile trajectory and find impact point
        """
        # Convert spherical to Cartesian coordinates for initial velocity
        elevation_rad = np.radians(elevation_angle)
        azimuth_rad = np.radians(azimuth_angle)
        
        vx0 = muzzle_velocity * np.cos(elevation_rad) * np.cos(azimuth_rad)
        vy0 = muzzle_velocity * np.cos(elevation_rad) * np.sin(azimuth_rad)
        vz0 = muzzle_velocity * np.sin(elevation_rad)
        
        # Initial state [x, y, z, vx, vy, vz]
        initial_state = np.array([0, 0, initial_altitude, vx0, vy0, vz0])
        
        # Create a local ground event that returns to the launch altitude (handles cases where cannon and target share altitude)
        def _ground_event(t, state, *args):
            # stop when z returns to the initial altitude (descending)
            return state[2] - initial_altitude

        _ground_event.terminal = True
        _ground_event.direction = -1

        # Solve trajectory
        solution = solve_ivp(
            self.projectile_dynamics,
            [0, max_time],
            initial_state,
            args=(muzzle_velocity, elevation_angle, azimuth_angle, wind_velocity),
            method='RK45',
            dense_output=True,
            events=[_ground_event],
            rtol=1e-8,
            atol=1e-10,
            max_step=0.1
        )
        
        return solution
    
    def calculate_impact(self, solution, target_distance):
        """Calculate impact point and error relative to target"""
        if not solution.success or len(solution.t_events[0]) == 0:
            return None, None, float('inf')
        
        impact_time = solution.t_events[0][0]
        impact_state = solution.sol(impact_time)
        impact_position = impact_state[:3]
        # Compute 2-D Euclidean distance from impact point to the target point (target_distance, 0)
        impact_xy = impact_position[:2]
        target_xy = np.array([target_distance, 0.0])    
        error_distance = float(np.linalg.norm(impact_xy - target_xy))

        return impact_position, impact_time, error_distance
    
    def firing_error(self, parameters, target_distance, initial_altitude, wind_velocity):
        """
        Objective function: minimize distance error to target
        parameters = [muzzle_velocity, elevation_angle, azimuth_angle]
        """
        muzzle_velocity, elevation_angle, azimuth_angle = parameters

        
        # Check constraints
        if (muzzle_velocity < 100 or muzzle_velocity > self.max_muzzle_velocity or
            elevation_angle < 1 or elevation_angle > 25 or
            azimuth_angle < -180 or azimuth_angle > 180):
            return float('inf')
        
        # Simulate trajectory
        solution = self.simulate_trajectory(
            muzzle_velocity, elevation_angle, azimuth_angle, 
            wind_velocity, initial_altitude, target_distance
        )

        impact_pos, impact_time, error = self.calculate_impact(solution, target_distance)

        # If no valid impact, return a large penalty
        if impact_pos is None or not np.isfinite(error):
            return 1e6

        # Return only the distance error. Do not penalize flight time here; we are not
        # trying to prefer the fastest solution, only the one that minimizes miss distance.
        return error
    
    def optimize_firing_solution(self, target_distance, initial_altitude, wind_velocity):
        """
        Find optimal firing parameters using hybrid optimization
        """
        # Parameter bounds for differential_evolution (list of (min,max) pairs)
        de_bounds = [(100, self.max_muzzle_velocity), (1, 25), (-180, 180)]
        
        # Phase 1: Global optimization
        # Use moderate DE settings to keep runs responsive; increase for production runs
        global_result = differential_evolution(
            self.firing_error,
            de_bounds,
            args=(target_distance, initial_altitude, wind_velocity),
            strategy='best1bin',
            maxiter=20,
            popsize=10,
            tol=0.1,
            seed=42,
            disp=False,
            polish=False
        )
        
        # Phase 2: Local refinement
        if global_result.success:
            local_result = minimize(
                self.firing_error,
                global_result.x,
                args=(target_distance, initial_altitude, wind_velocity),
                method='SLSQP',
                bounds=[(100, self.max_muzzle_velocity), (1, 25), (-180, 180)],
                options={'ftol': 1e-6, 'maxiter': 200}
            )
            
            if local_result.success:
                optimal_params = local_result.x
                final_error = local_result.fun
            else:
                optimal_params = global_result.x
                final_error = global_result.fun
        else:
            # Fallback to direct optimization
            optimal_params, final_error = self.direct_optimization(
                target_distance, initial_altitude, wind_velocity
            )
        
        return optimal_params, final_error
    
    def direct_optimization(self, target_distance, initial_altitude, wind_velocity):
        """Direct optimization using multiple starting points"""
        best_error = float('inf')
        best_params = None
        
        # Try different initial guesses
        initial_guesses = [
            [100, 20, 0],
            [200, 20, 0],
            [300, 20, 0],
            [400, 20, 0],   # Medium velocity, medium angle, straight
            [450, 15, 0],   # High velocity, low angle
            [350, 25, 0],   # Upper-bound angle candidate
            [400, 20, 10],  # With azimuth compensation
            [400, 20, -10]  # Opposite azimuth
        ]
        
        for guess in initial_guesses:
            result = minimize(
                self.firing_error,
                guess,
                args=(target_distance, initial_altitude, wind_velocity),
                method='SLSQP',
                bounds=[(100, self.max_muzzle_velocity), (1, 25), (-180, 180)],
                options={'ftol': 1e-6, 'maxiter': 100}
            )
            
            if result.success and result.fun < best_error:
                best_error = result.fun
                best_params = result.x
        
        if best_params is None:
            # Default fallback (respect new elevation cap)
            best_params = [400, 20, 0]
            best_error = self.firing_error(best_params, target_distance, initial_altitude, wind_velocity)
        
        # Run a quick coarse search to discover other feasible solutions (may find lower-velocity or faster-time hits)
        candidates = self.find_candidates_coarse(target_distance, initial_altitude, wind_velocity,
                                                 v_min=100, v_max=self.max_muzzle_velocity, v_step=50,
                                                 elev_min=1, elev_max=25, elev_step=2,
                                                 azim_choices=[-10, -5, 0, 5, 10],
                                                 error_tol=max(5.0, best_error + 1.0))

        if candidates:
            # choose candidate with smallest error, then smallest time
            candidates_sorted = sorted(candidates, key=lambda c: (c[4], c[3] if c[3] is not None else 1e9))
            v_c, elev_c, az_c, t_c, err_c = candidates_sorted[0]
            # Determine current best_time for a fair tie-break
            best_time = None
            if best_params is not None:
                sol_best = self.simulate_trajectory(best_params[0], best_params[1], best_params[2], wind_velocity, initial_altitude, target_distance)
                _, best_time, _ = self.calculate_impact(sol_best, target_distance)

            tol_err = 1e-6
            # Accept candidate if it improves error, or if equal within tolerance and its time is strictly shorter
            if err_c < best_error - tol_err:
                best_params = [v_c, elev_c, az_c]
                best_error = err_c
            elif abs(err_c - best_error) <= tol_err:
                if best_time is None and t_c is not None:
                    best_params = [v_c, elev_c, az_c]
                    best_error = err_c
                elif (t_c is not None and best_time is not None and t_c < best_time - 1e-6):
                    best_params = [v_c, elev_c, az_c]
                    best_error = err_c

        return best_params, best_error

    def find_candidates_coarse(self, target_distance, initial_altitude, wind_velocity,
                               v_min=100, v_max=None, v_step=50,
                               elev_min=1, elev_max=80, elev_step=2,
                               azim_choices=None, error_tol=20.0):
        """Coarse grid search to find candidate firing solutions quickly.

        Returns list of tuples (muzzle_velocity, elevation, azimuth, impact_time, error)
        """
        if v_max is None:
            v_max = self.max_muzzle_velocity
        if azim_choices is None:
            # If wind is negligible, only try azimuth = 0 to preserve left-right symmetry
            if np.linalg.norm(wind_velocity) < 1e-1:
                azim_choices = [0.0]
            else:
                azim_choices = [-10.0, -5.0, 0.0, 5.0, 10.0]

        candidates = []
        v_vals = list(range(int(v_min), int(v_max) + 1, int(v_step)))
        elev_vals = list(range(int(elev_min), int(elev_max) + 1, int(elev_step)))

        for v in v_vals:
            for elev in elev_vals:
                for az in azim_choices:
                    sol = self.simulate_trajectory(v, elev, az, wind_velocity, initial_altitude, target_distance)
                    impact_pos, impact_time, err = self.calculate_impact(sol, target_distance)
                    if impact_pos is not None and err <= error_tol:
                        candidates.append((v, elev, az, impact_time, err))

        return candidates
    
    def plot_trajectory_3d(self, muzzle_velocity, elevation_angle, azimuth_angle,
                          wind_velocity, initial_altitude, target_distance):
        """Plot 3D trajectory with wind effects, show lateral Y on side view, and acceleration time series."""
        # simulate
        solution = self.simulate_trajectory(
            muzzle_velocity, elevation_angle, azimuth_angle,
            wind_velocity, initial_altitude, target_distance
        )

        if not solution.success:
            print("Trajectory simulation failed")
            return None, None

        # Determine impact time (use event if available) and generate trajectory points up to impact
        if solution.t_events and len(solution.t_events) > 0 and len(solution.t_events[0]) > 0:
            end_time = float(solution.t_events[0][0])
        else:
            end_time = float(solution.t[-1])

        # avoid zero or negative end_time
        if end_time <= 0:
            end_time = float(solution.t[-1])

        t_eval = np.linspace(0, end_time, 500)
        trajectory = solution.sol(t_eval)
        x_traj = trajectory[0]
        y_traj = trajectory[1]
        z_traj = trajectory[2]
        vx_traj = trajectory[3]
        vy_traj = trajectory[4]
        vz_traj = trajectory[5]
        speed_traj = np.sqrt(vx_traj**2 + vy_traj**2 + vz_traj**2)

        # If there's essentially no wind, clean up tiny numerical y and vy drift for display and results
        # Use a slightly relaxed tolerance to catch tiny numerical noise from ODE solver
        y_tol = 1e-3  # meters
        vy_tol = 1e-3  # m/s
        if np.linalg.norm(wind_velocity) < 1e-1:
            if np.max(np.abs(y_traj)) < y_tol and np.max(np.abs(vy_traj)) < vy_tol:
                y_traj = np.zeros_like(y_traj)
                vy_traj = np.zeros_like(vy_traj)

        # Compute accelerations from aerodynamic drag + gravity at sampled points
        acc_traj = np.zeros((3, t_eval.size))
        for i in range(t_eval.size):
            vel = np.array([vx_traj[i], vy_traj[i], vz_traj[i]])
            a_drag = self.wind_force(vel, wind_velocity) / self.mass
            a_gravity = np.array([0.0, 0.0, -self.gravity])
            acc_traj[:, i] = a_drag + a_gravity

        acc_x_traj = acc_traj[0]
        acc_y_traj = acc_traj[1]
        acc_z_traj = acc_traj[2]
        acc_mag_traj = np.sqrt(acc_x_traj**2 + acc_y_traj**2 + acc_z_traj**2)

        impact_point, impact_time, error = self.calculate_impact(solution, target_distance)
        # If no wind and impact lateral displacement is tiny, set to exact zero for results/display
        try:
            if impact_point is not None and np.linalg.norm(wind_velocity) < 1e-1 and abs(impact_point[1]) < y_tol:
                impact_point[1] = 0.0
        except Exception:
            pass

        # Create 2x2 plot grid: 3D trajectory, top-down, side view, and acceleration vs time
        fig = plt.figure(figsize=(14, 9))

        # 3D trajectory
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        ax1.plot(x_traj, y_traj, z_traj, 'b-', linewidth=2, label='Trajectory')
        ax1.set_xlabel('X Distance (m)')
        ax1.set_ylabel('Y Distance (m)')
        ax1.set_zlabel('Altitude (m)')
        ax1.set_title('3D Trajectory')
        if impact_point is not None:
            ax1.plot([impact_point[0]], [impact_point[1]], [impact_point[2]], 'ro', markersize=8, label='Impact')
        ax1.plot([target_distance], [0.0], [initial_altitude], 'rx', markersize=10, label='Target')

        # Top-down view (X-Y plane)
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(x_traj, y_traj, 'g-', linewidth=2, label='Path')
        ax2.set_xlabel('X Distance (m)')
        ax2.set_ylabel('Y Distance (m)')
        ax2.set_title('Top-Down View')
        ax2.grid(True, alpha=0.3)
        ax2.axis('equal')
        pad_xy = max(np.ptp(x_traj), np.ptp(y_traj)) * 0.05 + 1.0
        ax2.set_xlim(min(x_traj.min(), 0) - pad_xy, max(x_traj.max(), target_distance) + pad_xy)
        ax2.set_ylim(y_traj.min() - pad_xy, y_traj.max() + pad_xy)
        if impact_point is not None:
            ax2.plot(impact_point[0], impact_point[1], 'ro', markersize=8, label='Impact')
            target_circle = plt.Circle((target_distance, 0), 10, color='red', fill=False, linestyle='--', label='Target')
            ax2.add_patch(target_circle)

        # Side view (X-Z plane) with lateral Y on secondary axis
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.plot(x_traj, z_traj, 'purple', linewidth=2, label='Altitude')
        ax3.set_xlabel('X Distance (m)')
        ax3.set_ylabel('Altitude (m)')
        ax3.set_title('Side View (with lateral Y)')
        ax3.grid(True, alpha=0.3)
        if impact_point is not None:
            ax3.plot(impact_point[0], impact_point[2], 'ro', markersize=8, label='Impact')

        pad_x = max(np.ptp(x_traj), 1.0) * 0.05 + 1.0
        pad_z = max(np.ptp(z_traj), 1.0) * 0.05 + 1.0
        ax3.set_xlim(min(x_traj.min(), 0) - pad_x, max(x_traj.max(), target_distance) + pad_x)
        ax3.set_ylim(min(z_traj.min(), initial_altitude) - pad_z, max(z_traj.max(), initial_altitude) + pad_z)

        try:
            ax3b = ax3.twinx()
            ax3b.plot(x_traj, y_traj, color='tab:green', linestyle='--', linewidth=1.5, label='Lateral Y')
            ax3b.set_ylabel('Lateral Y (m)', color='tab:green')
            ax3b.tick_params(axis='y', labelcolor='tab:green')
            if impact_point is not None:
                ax3b.plot(impact_point[0], impact_point[1], 'gx', markersize=8)
        except Exception:
            pass

        # Acceleration vs time plot (bottom-right)
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.plot(t_eval, acc_mag_traj, color='tab:red', linewidth=2, label='|a|')
        ax4.plot(t_eval, acc_x_traj, color='tab:blue', linewidth=1, linestyle='--', label='ax')
        ax4.plot(t_eval, acc_y_traj, color='tab:green', linewidth=1, linestyle='--', label='ay')
        ax4.plot(t_eval, acc_z_traj, color='tab:purple', linewidth=1, linestyle='--', label='az')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Acceleration (m/s²)')
        ax4.set_title('Acceleration vs Time')
        ax4.grid(True, alpha=0.3)
        ax4.legend()

        plt.tight_layout()
        try:
            ax1.legend()
            ax2.legend()
            ax3.legend()
        except Exception:
            pass
        plt.show()

        # Save kinematic data (position, velocity, speed, acceleration) to CSV for later analysis (optional)
        try:
            csv_path = f"trajectory_v{int(muzzle_velocity)}_e{int(elevation_angle)}_a{int(azimuth_angle)}.csv"
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['time_s', 'x_m', 'y_m', 'z_m',
                                 'vx_m_s', 'vy_m_s', 'vz_m_s', 'speed_m_s',
                                 'ax_m_s2', 'ay_m_s2', 'az_m_s2', 'a_mag_m_s2'])
                for ti, x, y, z, vx, vy, vz, sp, axc, ayc, azc, am in zip(
                        t_eval, x_traj, y_traj, z_traj, vx_traj, vy_traj, vz_traj,
                        speed_traj, acc_x_traj, acc_y_traj, acc_z_traj, acc_mag_traj):
                    writer.writerow([
                        f"{ti:.6f}", f"{x:.6f}", f"{y:.6f}", f"{z:.6f}",
                        f"{vx:.6f}", f"{vy:.6f}", f"{vz:.6f}", f"{sp:.6f}",
                        f"{axc:.6f}", f"{ayc:.6f}", f"{azc:.6f}", f"{am:.6f}"
                    ])
        except Exception:
            pass

        return impact_point, impact_time
    
    def analyze_wind_effect(self, target_distance, initial_altitude, base_wind):
        """Analyze how wind affects the firing solution"""
        wind_magnitudes = [0, 5, 10, 15, 20]  # m/s
        wind_directions = [0, 45, 90, 135, 180]  # degrees
        
        results = []
        
        print("WIND EFFECT ANALYSIS")
        print("=" * 60)
        
        for magnitude in wind_magnitudes:
            for direction in wind_directions:
                # Convert wind direction to velocity vector
                wind_rad = np.radians(direction)
                wind_velocity = np.array([
                    magnitude * np.cos(wind_rad),
                    magnitude * np.sin(wind_rad),
                    0
                ])
                
                # Find optimal solution (use faster direct optimization here)
                optimal_params, error = self.direct_optimization(
                    target_distance, initial_altitude, wind_velocity
                )
                
                muzzle_vel, elevation, azimuth = optimal_params
                
                results.append({
                    'wind_magnitude': magnitude,
                    'wind_direction': direction,
                    'muzzle_velocity': muzzle_vel,
                    'elevation': elevation,
                    'azimuth_correction': azimuth,
                    'error': error
                })
                
                print(f"Wind: {magnitude:2d} m/s @ {direction:3d}° | "
                      f"V: {muzzle_vel:5.1f} m/s | θ: {elevation:4.1f}° | "
                      f"Az: {azimuth:6.1f}° | Error: {error:5.1f} m")
        
        return results
    
    def generate_firing_table(self, distances, wind_conditions):
        """Generate firing table for artillery officer"""
        initial_altitude = 500  # Both cannon and target at same altitude
        
        print("ARTILLERY FIRING TABLE")
        print("=" * 80)
        print("Range (m) | Wind (m/s) | Wind Dir (°) | Velocity (m/s) | Elevation (°) | Azimuth (°) | Error (m)")
        print("-" * 80)
        
        firing_table = []
        
        for distance in distances:
            for wind_mag, wind_dir in wind_conditions:
                # Convert wind to vector
                wind_rad = np.radians(wind_dir)
                wind_velocity = np.array([
                    wind_mag * np.cos(wind_rad),
                    wind_mag * np.sin(wind_rad),
                    0
                ])
                
                # Calculate solution (use faster direct optimization for table generation)
                optimal_params, error = self.direct_optimization(
                    distance, initial_altitude, wind_velocity
                )
                
                muzzle_vel, elevation, azimuth = optimal_params
                
                firing_table.append({
                    'distance': distance,
                    'wind_magnitude': wind_mag,
                    'wind_direction': wind_dir,
                    'muzzle_velocity': muzzle_vel,
                    'elevation': elevation,
                    'azimuth': azimuth,
                    'error': error
                })
                
                print(f"{distance:8.0f} | {wind_mag:10.1f} | {wind_dir:12.0f} | "
                      f"{muzzle_vel:14.1f} | {elevation:12.1f} | {azimuth:10.1f} | "
                      f"{error:8.1f}")
        
        return firing_table

def main():
    """Main execution function"""
    # Initialize artillery solver
    artillery = ArtillerySolver3D()
    
    # Common parameters
    initial_altitude = 500  # Both cannon and target at 500m altitude
    
    print("3D ARTILLERY FIRING SOLUTION WITH WIND EFFECTS")
    print("=" * 70)
    
    # Test case 1: 1200m target, no wind
    print("\nCASE 1: 1200m TARGET, NO WIND")
    target_distance = 1200
    wind_velocity = np.array([0, 0, 0])  # No wind
    
    # use faster direct optimization for the initial quick run
    optimal_params, error = artillery.direct_optimization(
        target_distance, initial_altitude, wind_velocity
    )
    
    muzzle_vel, elevation, azimuth = optimal_params
    
    print(f"Optimal Solution:")
    print(f"  Muzzle Velocity: {muzzle_vel:.1f} m/s")
    print(f"  Elevation Angle: {elevation:.1f}°")
    print(f"  Azimuth Angle: {azimuth:.1f}°")
    print(f"  Expected Error: {error:.2f} m")
    
    # Plot trajectory
    impact_point, impact_time = artillery.plot_trajectory_3d(
        muzzle_vel, elevation, azimuth, wind_velocity, initial_altitude, target_distance
    )
    
    if impact_point is not None:
        horizontal_distance = np.linalg.norm(impact_point[:2])
        print(f"Impact Point: ({impact_point[0]:.1f}, {impact_point[1]:.1f}, {impact_point[2]:.1f})")
        print(f"Horizontal Distance: {horizontal_distance:.1f} m")
        print(f"Flight Time: {impact_time:.2f} s")
    
    # Test case 2: 1200m target with crosswind
    print("\nCASE 2: 1200m TARGET WITH CROSSWIND (10 m/s from West)")
    wind_velocity = np.array([0, 20, 0])  # 10 m/s from West
    
    optimal_params, error = artillery.direct_optimization(
        target_distance, initial_altitude, wind_velocity
    )
    
    muzzle_vel, elevation, azimuth = optimal_params
    
    print(f"Optimal Solution:")
    print(f"  Muzzle Velocity: {muzzle_vel:.1f} m/s")
    print(f"  Elevation Angle: {elevation:.1f}°")
    print(f"  Azimuth Angle: {azimuth:.1f}° (wind compensation)")
    print(f"  Expected Error: {error:.2f} m")
    
    # Plot trajectory with wind
    impact_point, impact_time = artillery.plot_trajectory_3d(
        muzzle_vel, elevation, azimuth, wind_velocity, initial_altitude, target_distance
    )
    
    # # Wind effect analysis
    # print("\nCOMPREHENSIVE WIND EFFECT ANALYSIS")
    # wind_analysis = artillery.analyze_wind_effect(1200, initial_altitude, 10)
    
    # Generate firing table for practical use
    print("\nPRACTICAL FIRING TABLE")
    distances = [1000, 1100, 1200, 1300, 1400, 1500]
    wind_conditions = [(0, 0), (5, 90), (10, 90), (5, 270), (10, 270)]  # (magnitude, direction)
    
    firing_table = artillery.generate_firing_table(distances, wind_conditions)
    
    # Performance analysis across ranges
    print("\nRANGE PERFORMANCE SUMMARY")
    print("=" * 50)
    print("Range (m) | Avg Velocity (m/s) | Avg Elevation (°) | Avg Error (m)")
    print("-" * 50)
    
    for distance in distances:
        distance_data = [item for item in firing_table if item['distance'] == distance]
        avg_velocity = np.mean([item['muzzle_velocity'] for item in distance_data])
        avg_elevation = np.mean([item['elevation'] for item in distance_data])
        avg_error = np.mean([item['error'] for item in distance_data])
        
        print(f"{distance:8.0f} | {avg_velocity:17.1f} | {avg_elevation:16.1f} | {avg_error:12.1f}")

if __name__ == "__main__":
    main()