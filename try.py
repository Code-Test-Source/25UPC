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
        self.drag_coeff = 0.47             # Sphere drag coefficient
        self.air_density = 1.2             # Air density [kg/m³]
        self.gravity = 9.8                 # Gravity [m/s²]
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
        """Calculate wind-induced force on projectile"""
        relative_velocity = velocity - wind_velocity
        speed = np.linalg.norm(relative_velocity)
        
        if speed < 1e-10:
            return np.zeros(3)
            
        # Drag force magnitude
        drag_magnitude = self.k_drag * speed**2
        
        # Direction of drag force (opposite to relative velocity)
        drag_direction = -relative_velocity / speed
        
        return drag_magnitude * drag_direction
    
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
            elevation_angle < 1 or elevation_angle > 80 or
            azimuth_angle < -180 or azimuth_angle > 180):
            return float('inf')
        
        # Simulate trajectory
        solution = self.simulate_trajectory(
            muzzle_velocity, elevation_angle, azimuth_angle, 
            wind_velocity, initial_altitude, target_distance
        )
        
        _, _, error = self.calculate_impact(solution, target_distance)
        
        return error
    
    def optimize_firing_solution(self, target_distance, initial_altitude, wind_velocity):
        """
        Find optimal firing parameters using hybrid optimization
        """
        # Parameter bounds for differential_evolution (list of (min,max) pairs)
        de_bounds = [(100, self.max_muzzle_velocity), (1, 80), (-180, 180)]
        
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
                bounds=[(100, self.max_muzzle_velocity), (1, 80), (-180, 180)],
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
            [400, 30, 0],   # Medium velocity, medium angle, straight
            [450, 20, 0],   # High velocity, low angle
            [350, 45, 0],   # Lower velocity, high angle
            [400, 25, 10],  # With azimuth compensation
            [400, 25, -10]  # Opposite azimuth
        ]
        
        for guess in initial_guesses:
            result = minimize(
                self.firing_error,
                guess,
                args=(target_distance, initial_altitude, wind_velocity),
                method='SLSQP',
                bounds=[(100, self.max_muzzle_velocity), (1, 80), (-180, 180)],
                options={'ftol': 1e-6, 'maxiter': 100}
            )
            
            if result.success and result.fun < best_error:
                best_error = result.fun
                best_params = result.x
        
        if best_params is None:
            # Default fallback
            best_params = [400, 30, 0]
            best_error = self.firing_error(best_params, target_distance, initial_altitude, wind_velocity)
        
        return best_params, best_error
    
    def plot_trajectory_3d(self, muzzle_velocity, elevation_angle, azimuth_angle,
                          wind_velocity, initial_altitude, target_distance):
        """Plot 3D trajectory with wind effects"""
        solution = self.simulate_trajectory(
            muzzle_velocity, elevation_angle, azimuth_angle,
            wind_velocity, initial_altitude, target_distance
        )
        
        if not solution.success:
            print("Trajectory simulation failed")
            return None, None
        
        # Generate trajectory points
        t_eval = np.linspace(0, solution.t[-1], 500)
        trajectory = solution.sol(t_eval)
        x_traj, y_traj, z_traj = trajectory[0], trajectory[1], trajectory[2]
        
        # Calculate impact point
        impact_point, impact_time, error = self.calculate_impact(solution, target_distance)
        
        # Create 3D plot
        fig = plt.figure(figsize=(15, 5))
        
        # 3D trajectory
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.plot(x_traj, y_traj, z_traj, 'b-', linewidth=2, label='Trajectory')
        ax1.set_xlabel('X Distance (m)')
        ax1.set_ylabel('Y Distance (m)')
        ax1.set_zlabel('Altitude (m)')
        ax1.set_title('3D Trajectory')
        
        if impact_point is not None:
            ax1.plot([impact_point[0]], [impact_point[1]], [impact_point[2]], 
                    'ro', markersize=8, label='Impact')
        
        # Top-down view (X-Y plane)
        ax2 = fig.add_subplot(132)
        ax2.plot(x_traj, y_traj, 'g-', linewidth=2, label='Path')
        ax2.set_xlabel('X Distance (m)')
        ax2.set_ylabel('Y Distance (m)')
        ax2.set_title('Top-Down View')
        ax2.grid(True, alpha=0.3)
        ax2.axis('equal')
        
        if impact_point is not None:
            ax2.plot(impact_point[0], impact_point[1], 'ro', markersize=8, label='Impact')
            # Target circle
            target_circle = plt.Circle((target_distance, 0), 10, color='red', 
                                     fill=False, linestyle='--', label='Target')
            ax2.add_patch(target_circle)
        
        # Side view (X-Z plane)
        ax3 = fig.add_subplot(133)
        ax3.plot(x_traj, z_traj, 'purple', linewidth=2, label='Altitude')
        ax3.set_xlabel('X Distance (m)')
        ax3.set_ylabel('Altitude (m)')
        ax3.set_title('Side View')
        ax3.grid(True, alpha=0.3)
        
        if impact_point is not None:
            ax3.plot(impact_point[0], impact_point[2], 'ro', markersize=8, label='Impact')
        
        plt.tight_layout()
        plt.legend()
        plt.show()
        
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
    wind_velocity = np.array([10, 0, 0])  # 10 m/s from West
    
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
    
    # Wind effect analysis
    print("\nCOMPREHENSIVE WIND EFFECT ANALYSIS")
    wind_analysis = artillery.analyze_wind_effect(1200, initial_altitude, 10)
    
    # Generate firing table for practical use
    print("\nPRACTICAL FIRING TABLE")
    distances = [800, 1000, 1200, 1400, 1600]
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