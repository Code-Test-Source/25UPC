import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, HuberRegressor, RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error
import os
import glob

class ArtilleryFormulaFitter:
    def __init__(self):
        # Choose robust fitting method: 'huber' (default) or 'ransac'
        self.robust_method = 'huber'
        self.robust_alpha = 0.01
        self.ransac_residual_threshold = None
        # Elevation peak weighting: center (deg) and sigma (deg) for Gaussian weights
        # If center is None, we'll use the median elevation from the training targets.
        self.elevation_peak_center = None
        self.elevation_peak_sigma = 1.5

        self.velocity_model = None
        self.elevation_model = None
        self.azimuth_model = None
        self.density_correction_model = None
        
    def load_firing_table_data(self, file_pattern="firing_table_density_*.csv"):
        """Load firing table data for all densities.

        Returns a combined pandas DataFrame with an added 'density' column
        extracted from each filename.
        """
        all_data = []
        
        # 查找所有匹配的文件
        files = glob.glob(file_pattern)
        if not files:
            raise FileNotFoundError(f"No files found matching pattern: {file_pattern}")
        
        print(f"Found {len(files)} data files:")
        for file in files:
            # 从文件名提取密度信息
            density = float(file.split('_')[-1].replace('.csv', ''))
            print(f"  {file} -> density: {density}")
            
            df = pd.read_csv(file)
            df['density'] = density  # 添加密度列
            all_data.append(df)
        
        # 合并所有数据
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"Loaded {len(combined_df)} total data points")
        return combined_df
    
    def prepare_features(self, df):
        """Prepare feature matrix, including trig transforms for wind direction."""
        X = df[['distance', 'wind_magnitude', 'wind_direction', 'density']].copy()
        
        # 对风向进行三角函数变换，捕捉周期性
        X['wind_dir_sin'] = np.sin(np.radians(X['wind_direction']))
        X['wind_dir_cos'] = np.cos(np.radians(X['wind_direction']))
        
        # 添加交互特征
        X['distance_wind_speed'] = X['distance'] * X['wind_magnitude']
        X['wind_speed_dir_sin'] = X['wind_magnitude'] * X['wind_dir_sin']
        X['wind_speed_dir_cos'] = X['wind_magnitude'] * X['wind_dir_cos']
        
        # 密度相关特征
        X['density_inv_sqrt'] = 1.0 / np.sqrt(X['density'])
        X['distance_density'] = X['distance'] * X['density_inv_sqrt']
        
        return X
    
    def fit_velocity_formula(self, X, y_velocity):
        """Fit a simple linear formula for muzzle velocity."""
        # 选择最相关的特征
        velocity_features = ['distance', 'wind_magnitude', 'wind_dir_cos', 'density_inv_sqrt']
        X_velocity = X[velocity_features]

        # Choose robust fitting strategy: RANSAC focuses on the largest consensus set
        # (the peak of the distribution); Huber reduces outlier influence while
        # keeping smooth loss near zero residuals.
        if self.robust_method == 'ransac':
            base = LinearRegression()
            model = RANSACRegressor(base_estimator=base, residual_threshold=self.ransac_residual_threshold)
            model.fit(X_velocity, y_velocity)
            coef = model.estimator_.coef_
            intercept = model.estimator_.intercept_
        else:
            model = HuberRegressor(alpha=self.robust_alpha, max_iter=1000)
            model.fit(X_velocity, y_velocity)
            coef = model.coef_
            intercept = model.intercept_

        self.velocity_model = model

        # compute fit quality
        y_pred = self.velocity_model.predict(X_velocity)
        r2 = r2_score(y_velocity, y_pred)
        mae = mean_absolute_error(y_velocity, y_pred)

        method_name = 'RANSAC' if self.robust_method == 'ransac' else 'Huber'
        print(f"Velocity model R²: {r2:.4f}, MAE: {mae:.2f} m/s ({method_name})")

        return {
            'coefficients': coef,
            'intercept': intercept,
            'features': velocity_features,
            'r2': r2,
            'mae': mae
        }
    
    def fit_elevation_formula(self, X, y_elevation):
        """Fit a simple linear formula for elevation angle."""
        # 仰角受距离影响最大，风速影响较小
        elevation_features = ['distance', 'wind_magnitude', 'wind_dir_sin']
        X_elevation = X[elevation_features]

        # If the user wants to focus strictly on a narrow elevation band, filter to that range.
        # New attributes (can be set by caller): elevation_focus_min, elevation_focus_max
        focus_min = getattr(self, 'elevation_focus_min', None)
        focus_max = getattr(self, 'elevation_focus_max', None)

        if focus_min is not None and focus_max is not None:
            # build mask on the target elevations
            mask = (y_elevation >= float(focus_min)) & (y_elevation <= float(focus_max))
            n_in = int(np.sum(mask))
            if n_in < 10:
                print(f"Warning: only {n_in} elevation samples in [{focus_min},{focus_max}]°, falling back to weighted fit.")
            else:
                # restrict training data to the focus band
                X_elev_fit = X_elevation.loc[mask].reset_index(drop=True)
                y_elev_fit = y_elevation.loc[mask].reset_index(drop=True)

                if self.robust_method == 'ransac':
                    base = LinearRegression()
                    model = RANSACRegressor(base_estimator=base, residual_threshold=self.ransac_residual_threshold)
                    model.fit(X_elev_fit, y_elev_fit)
                    coef = model.estimator_.coef_
                    intercept = model.estimator_.intercept_
                else:
                    model = HuberRegressor(alpha=self.robust_alpha, max_iter=1000)
                    model.fit(X_elev_fit, y_elev_fit)
                    coef = model.coef_
                    intercept = model.intercept_

                self.elevation_model = model

                y_pred = self.elevation_model.predict(X_elev_fit)
                r2 = r2_score(y_elev_fit, y_pred)
                mae = mean_absolute_error(y_elev_fit, y_pred)

                method_name = 'RANSAC' if self.robust_method == 'ransac' else 'Huber'
                print(f"Elevation model (focused {focus_min}-{focus_max}°) R²: {r2:.4f}, MAE: {mae:.2f}° ({method_name}), n={n_in}")

                return {
                    'coefficients': coef,
                    'intercept': intercept,
                    'features': elevation_features,
                    'r2': r2,
                    'mae': mae
                }

        # If not focusing or not enough samples, fall back to weighted fit over all data
        # Build sample weights that emphasize the peak (most common) elevation values.
        center = self.elevation_peak_center if self.elevation_peak_center is not None else np.median(y_elevation)
        sigma = max(0.1, float(self.elevation_peak_sigma))
        # Gaussian weights centered on the peak; points near center get weight~1, far points downweighted
        weights = np.exp(-0.5 * ((y_elevation - center) / sigma) ** 2)
        # normalize weights to have mean=1 to keep loss scale similar
        weights = weights / np.mean(weights)

        if self.robust_method == 'ransac':
            base = LinearRegression()
            model = RANSACRegressor(base_estimator=base, residual_threshold=self.ransac_residual_threshold)
            model.fit(X_elevation, y_elevation, sample_weight=weights)
            coef = model.estimator_.coef_
            intercept = model.estimator_.intercept_
        else:
            model = HuberRegressor(alpha=self.robust_alpha, max_iter=1000)
            model.fit(X_elevation, y_elevation, sample_weight=weights)
            coef = model.coef_
            intercept = model.intercept_

        self.elevation_model = model

        y_pred = self.elevation_model.predict(X_elevation)
        r2 = r2_score(y_elevation, y_pred)
        mae = mean_absolute_error(y_elevation, y_pred)

        method_name = 'RANSAC' if self.robust_method == 'ransac' else 'Huber'
        print(f"Elevation model R²: {r2:.4f}, MAE: {mae:.2f}° ({method_name})")

        return {
            'coefficients': coef,
            'intercept': intercept,
            'features': elevation_features,
            'r2': r2,
            'mae': mae
        }
    
    def fit_azimuth_formula(self, X, y_azimuth):
        """Fit a simple linear formula for azimuth correction."""
        # 方位角主要由横风分量决定
        azimuth_features = ['wind_speed_dir_sin']
        X_azimuth = X[azimuth_features]

        if self.robust_method == 'ransac':
            base = LinearRegression()
            model = RANSACRegressor(base_estimator=base, residual_threshold=self.ransac_residual_threshold)
            model.fit(X_azimuth, y_azimuth)
            coef = model.estimator_.coef_
            intercept = model.estimator_.intercept_
        else:
            model = HuberRegressor(alpha=self.robust_alpha, max_iter=1000)
            model.fit(X_azimuth, y_azimuth)
            coef = model.coef_
            intercept = model.intercept_

        self.azimuth_model = model

        y_pred = self.azimuth_model.predict(X_azimuth)
        r2 = r2_score(y_azimuth, y_pred)
        mae = mean_absolute_error(y_azimuth, y_pred)

        method_name = 'RANSAC' if self.robust_method == 'ransac' else 'Huber'
        print(f"Azimuth model R²: {r2:.4f}, MAE: {mae:.2f}° ({method_name})")

        return {
            'coefficients': coef,
            'intercept': intercept,
            'features': azimuth_features,
            'r2': r2,
            'mae': mae
        }
    
    def fit_all_formulas(self, df):
        """Fit all empirical formulas (velocity, elevation, azimuth)."""
        print("Preparing features...")
        X = self.prepare_features(df)
        
        print("\nFitting velocity formula...")
        velocity_result = self.fit_velocity_formula(X, df['muzzle_velocity'])
        
        print("\nFitting elevation formula...")
        elevation_result = self.fit_elevation_formula(X, df['elevation'])
        
        print("\nFitting azimuth formula...")
        azimuth_result = self.fit_azimuth_formula(X, df['azimuth'])
        
        return {
            'velocity': velocity_result,
            'elevation': elevation_result,
            'azimuth': azimuth_result
        }
    
    def generate_manual_formulas(self, fit_results):
        """Generate human-readable/manual calculation formulas from fit results."""
        v = fit_results['velocity']
        e = fit_results['elevation']
        a = fit_results['azimuth']
        
        # 提取系数
        v_coef = v['coefficients']
        v_int = v['intercept']
        
        e_coef = e['coefficients']
        e_int = e['intercept']
        
        a_coef = a['coefficients']
        a_int = a['intercept']
        
        manual_formulas = {
            'velocity': f"v = {v_int:.1f} + {v_coef[0]:.4f}×距离 + {v_coef[1]:.4f}×风速 + {v_coef[2]:.4f}×cos(风向) + {v_coef[3]:.4f}/√密度",
            'elevation': f"θ = {e_int:.1f} + {e_coef[0]:.4f}×距离 + {e_coef[1]:.4f}×风速 + {e_coef[2]:.4f}×sin(风向)",
            'azimuth': f"α = {a_int:.3f} + {a_coef[0]:.4f}×风速×sin(风向)",
            
            'simplified_velocity': f"v = ({v_int:.0f} + {v_coef[0]:.3f}×D + {v_coef[1]:.3f}×W + {v_coef[2]:.3f}×W×cos(θ_w)) × (1/√ρ)^{v_coef[3]:.3f}",
            'simplified_elevation': f"θ = {e_int:.1f} + {e_coef[0]:.4f}×D + {e_coef[1]:.3f}×W + {e_coef[2]:.3f}×W×sin(θ_w)",
            'simplified_azimuth': f"α = {a_coef[0]:.3f} × W × sin(θ_w)",
            
            'coefficients': {
                'velocity_base': v_int,
                'velocity_distance_coef': v_coef[0],
                'velocity_wind_coef': v_coef[1],
                'velocity_wind_dir_coef': v_coef[2],
                'velocity_density_coef': v_coef[3],
                
                'elevation_base': e_int,
                'elevation_distance_coef': e_coef[0],
                'elevation_wind_coef': e_coef[1],
                'elevation_wind_dir_coef': e_coef[2],
                
                'azimuth_base': a_int,
                'azimuth_wind_coef': a_coef[0]
            }
        }
        
        return manual_formulas
    
    def calculate_with_formulas(self, distance, wind_speed, wind_dir, density=1.2):
        """Calculate velocity/elevation/azimuth using the fitted formulas."""
        if not all([self.velocity_model, self.elevation_model, self.azimuth_model]):
            raise ValueError("Models not fitted yet. Call fit_all_formulas first.")
        
        # 准备特征
        wind_dir_rad = np.radians(wind_dir)
        features_dict = {
            'distance': distance,
            'wind_magnitude': wind_speed,
            'wind_direction': wind_dir,
            'density': density,
            'wind_dir_sin': np.sin(wind_dir_rad),
            'wind_dir_cos': np.cos(wind_dir_rad),
            'wind_speed_dir_sin': wind_speed * np.sin(wind_dir_rad),
            'density_inv_sqrt': 1.0 / np.sqrt(density)
        }
        
        # 速度计算 - pass a DataFrame with the original feature names to avoid sklearn warnings
        velocity_features = ['distance', 'wind_magnitude', 'wind_dir_cos', 'density_inv_sqrt']
        X_velocity = pd.DataFrame([{f: features_dict[f] for f in velocity_features}])
        velocity = self.velocity_model.predict(X_velocity)[0]

        # 仰角计算
        elevation_features = ['distance', 'wind_magnitude', 'wind_dir_sin']
        X_elevation = pd.DataFrame([{f: features_dict[f] for f in elevation_features}])
        elevation = self.elevation_model.predict(X_elevation)[0]

        # 方位角计算
        azimuth_features = ['wind_speed_dir_sin']
        X_azimuth = pd.DataFrame([{f: features_dict[f] for f in azimuth_features}])
        azimuth = self.azimuth_model.predict(X_azimuth)[0]

        return {
            'velocity': max(100, min(450, velocity)),  # 限制在合理范围
            'elevation': max(10, min(25, elevation)),  # 限制在10-25度
            'azimuth': azimuth
        }
    
    def plot_fitting_results(self, df, fit_results):
        """Plot fitting diagnostics as two separate figures:
        - Figure 1: predicted vs actual scatter plots for velocity, elevation, azimuth
        - Figure 2: error histograms for each quantity
        Returns the same error statistics dict as before.
        """
        X = self.prepare_features(df)

        # Create and save six separate JPGs (three scatter plots and three histograms)
        # Predictions
        y_velocity_pred = self.velocity_model.predict(X[fit_results['velocity']['features']])
        elev_features = fit_results['elevation']['features']
        y_elevation_pred_full = self.elevation_model.predict(X[elev_features])
        focus_min = getattr(self, 'elevation_focus_min', None)
        focus_max = getattr(self, 'elevation_focus_max', None)
        y_azimuth_pred = self.azimuth_model.predict(X[fit_results['azimuth']['features']])

        # --- Scatter: Velocity ---
        fig_v, ax_v = plt.subplots(figsize=(6, 5))
        ax_v.scatter(df['muzzle_velocity'], y_velocity_pred, alpha=0.6)
        ax_v.plot([df['muzzle_velocity'].min(), df['muzzle_velocity'].max()],
                  [df['muzzle_velocity'].min(), df['muzzle_velocity'].max()], 'r--')
        ax_v.set_xlabel('Actual velocity (m/s)')
        ax_v.set_ylabel('Predicted velocity (m/s)')
        ax_v.set_title(f'Velocity fit (R²={fit_results["velocity"]["r2"]:.3f})')
        ax_v.grid(True, alpha=0.3)
        fig_v.tight_layout()
        fig_v.savefig('fitting_velocity_scatter.jpg', dpi=200)
        plt.close(fig_v)

        # --- Scatter: Elevation (focused if configured) ---
        fig_e, ax_e = plt.subplots(figsize=(6, 5))
        if focus_min is not None and focus_max is not None:
            mask = (df['elevation'] >= float(focus_min)) & (df['elevation'] <= float(focus_max))
            n_mask = int(mask.sum())
            if n_mask > 0:
                y_elev_series = pd.Series(y_elevation_pred_full, index=df.index)
                ax_e.scatter(df.loc[mask, 'elevation'], y_elev_series.loc[mask], alpha=0.7)
                ax_e.plot([focus_min, focus_max], [focus_min, focus_max], 'r--')
                ax_e.set_title(f'Elevation fit (focused {focus_min}-{focus_max}°, n={n_mask})')
            else:
                ax_e.scatter(df['elevation'], y_elevation_pred_full, alpha=0.6)
                ax_e.plot([df['elevation'].min(), df['elevation'].max()],
                          [df['elevation'].min(), df['elevation'].max()], 'r--')
                ax_e.set_title(f'Elevation fit (R²={fit_results["elevation"]["r2"]:.3f})')
        else:
            ax_e.scatter(df['elevation'], y_elevation_pred_full, alpha=0.6)
            ax_e.plot([df['elevation'].min(), df['elevation'].max()],
                      [df['elevation'].min(), df['elevation'].max()], 'r--')
            ax_e.set_title(f'Elevation fit (R²={fit_results["elevation"]["r2"]:.3f})')

        ax_e.set_xlabel('Actual elevation (°)')
        ax_e.set_ylabel('Predicted elevation (°)')
        ax_e.grid(True, alpha=0.3)
        fig_e.tight_layout()
        fig_e.savefig('fitting_elevation_scatter.jpg', dpi=200)
        plt.close(fig_e)

        # --- Scatter: Azimuth ---
        fig_a, ax_a = plt.subplots(figsize=(6, 5))
        ax_a.scatter(df['azimuth'], y_azimuth_pred, alpha=0.6)
        ax_a.plot([df['azimuth'].min(), df['azimuth'].max()],
                  [df['azimuth'].min(), df['azimuth'].max()], 'r--')
        ax_a.set_xlabel('Actual azimuth (°)')
        ax_a.set_ylabel('Predicted azimuth (°)')
        ax_a.set_title(f'Azimuth fit (R²={fit_results["azimuth"]["r2"]:.3f})')
        ax_a.grid(True, alpha=0.3)
        fig_a.tight_layout()
        fig_a.savefig('fitting_azimuth_scatter.jpg', dpi=200)
        plt.close(fig_a)

        # --- Histogram: Velocity error ---
        velocity_error = y_velocity_pred - df['muzzle_velocity']
        fig_vh, ax_vh = plt.subplots(figsize=(6, 5))
        ax_vh.hist(velocity_error, bins=30, alpha=0.7)
        ax_vh.axvline(velocity_error.mean(), color='red', linestyle='--',
                      label=f'Mean error: {velocity_error.mean():.2f} m/s')
        ax_vh.set_xlabel('Velocity error (m/s)')
        ax_vh.set_ylabel('Count')
        ax_vh.set_title('Velocity error distribution')
        ax_vh.legend()
        ax_vh.grid(True, alpha=0.3)
        fig_vh.tight_layout()
        fig_vh.savefig('fitting_velocity_error.jpg', dpi=200)
        plt.close(fig_vh)

        # --- Histogram: Elevation error (focused if configured) ---
        y_elev_series = pd.Series(y_elevation_pred_full, index=df.index)
        fig_eh, ax_eh = plt.subplots(figsize=(6, 5))
        if focus_min is not None and focus_max is not None and (df['elevation'] >= float(focus_min)).any():
            mask = (df['elevation'] >= float(focus_min)) & (df['elevation'] <= float(focus_max))
            elevation_error = y_elev_series.loc[mask] - df.loc[mask, 'elevation']
            ax_eh.hist(elevation_error, bins=30, alpha=0.7)
            ax_eh.axvline(elevation_error.mean(), color='red', linestyle='--',
                           label=f'Mean error (focus): {elevation_error.mean():.2f}°')
            ax_eh.set_title(f'Elevation error (focused {focus_min}-{focus_max}°, n={int(mask.sum())})')
        else:
            elevation_error = y_elev_series - df['elevation']
            ax_eh.hist(elevation_error, bins=30, alpha=0.7)
            ax_eh.axvline(elevation_error.mean(), color='red', linestyle='--',
                           label=f'Mean error: {elevation_error.mean():.2f}°')
        ax_eh.set_xlabel('Elevation error (°)')
        ax_eh.set_ylabel('Count')
        ax_eh.set_title('Elevation error distribution')
        ax_eh.legend()
        ax_eh.grid(True, alpha=0.3)
        fig_eh.tight_layout()
        fig_eh.savefig('fitting_elevation_error.jpg', dpi=200)
        plt.close(fig_eh)

        # --- Histogram: Azimuth error ---
        azimuth_error = y_azimuth_pred - df['azimuth']
        fig_ah, ax_ah = plt.subplots(figsize=(6, 5))
        ax_ah.hist(azimuth_error, bins=30, alpha=0.7)
        ax_ah.axvline(azimuth_error.mean(), color='red', linestyle='--',
                       label=f'Mean error: {azimuth_error.mean():.2f}°')
        ax_ah.set_xlabel('Azimuth error (°)')
        ax_ah.set_ylabel('Count')
        ax_ah.set_title('Azimuth error distribution')
        ax_ah.legend()
        ax_ah.grid(True, alpha=0.3)
        fig_ah.tight_layout()
        fig_ah.savefig('fitting_azimuth_error.jpg', dpi=200)
        plt.close(fig_ah)

        return {
            'velocity_error_stats': {
                'mean': velocity_error.mean(),
                'std': velocity_error.std(),
                'max': float(np.max(np.abs(velocity_error)))
            },
            'elevation_error_stats': {
                'mean': elevation_error.mean(),
                'std': elevation_error.std(),
                'max': float(np.max(np.abs(elevation_error)))
            },
            'azimuth_error_stats': {
                'mean': azimuth_error.mean(),
                'std': azimuth_error.std(),
                'max': float(np.max(np.abs(azimuth_error)))
            }
        }

def main():
    """Main: fit empirical formulas from firing table CSVs."""
    fitter = ArtilleryFormulaFitter()

    try:
        # Load data
        print("Loading firing table data...")
        df = fitter.load_firing_table_data()

        # Fit formulas
        print("\nFitting empirical formulas...")

        # --- IMPORTANT: by default focus elevation fitting on 23-25 degrees ---
        # This enforces the requested behavior: train elevation model only on
        # samples whose elevation lies in [23, 25]. You can change or disable
        # this by editing these attributes before running.
        fitter.elevation_focus_min = 23.0
        fitter.elevation_focus_max = 25.0
        fitter.elevation_peak_sigma = 1.0
        # Keep Huber as default robust method; switch to 'ransac' for consensus-only
        fitter.robust_method = getattr(fitter, 'robust_method', 'huber')

        print(f"Focusing elevation fit on range [{fitter.elevation_focus_min}, {fitter.elevation_focus_max}]°")
        fit_results = fitter.fit_all_formulas(df)

        # Generate manual formulas
        print("\nGenerating manual calculation formulas...")
        formulas = fitter.generate_manual_formulas(fit_results)

        # Display fitted formulas
        print("\n" + "="*60)
        print("FITTED ARTILLERY FORMULAS")
        print("="*60)

        print("\nVELOCITY FORMULA:")
        print(formulas['velocity'])
        print(f"Simplified: {formulas['simplified_velocity']}")

        print("\nELEVATION FORMULA:")
        print(formulas['elevation'])
        print(f"Simplified: {formulas['simplified_elevation']}")

        print("\nAZIMUTH FORMULA:")
        print(formulas['azimuth'])
        print(f"Simplified: {formulas['simplified_azimuth']}")

        # Show numerical coefficients
        print("\nNUMERICAL COEFFICIENTS:")
        coeffs = formulas['coefficients']
        print(f"Velocity: base={coeffs['velocity_base']:.2f}, distance_coef={coeffs['velocity_distance_coef']:.4f}")
        print(f"          wind_coef={coeffs['velocity_wind_coef']:.4f}, wind_dir_coef={coeffs['velocity_wind_dir_coef']:.4f}")
        print(f"          density_coef={coeffs['velocity_density_coef']:.4f}")

        print(f"Elevation: base={coeffs['elevation_base']:.2f}, distance_coef={coeffs['elevation_distance_coef']:.4f}")
        print(f"           wind_coef={coeffs['elevation_wind_coef']:.4f}, wind_dir_coef={coeffs['elevation_wind_dir_coef']:.4f}")

        print(f"Azimuth: base={coeffs['azimuth_base']:.4f}, wind_coef={coeffs['azimuth_wind_coef']:.4f}")

        # Validate formulas with sample cases
        print("\n" + "="*60)
        print("FORMULA VALIDATION")
        print("="*60)

        test_cases = [
            (1200, 10, 90, 1.2),
            (1500, 5, 0, 1.2),
            (1000, 15, 180, 1.1),
            (1300, 8, 270, 1.3)
        ]

        print("\nTest calculations:")
        print("Distance(m) | Wind speed(m/s) | Wind dir(°) | Density | Velocity(m/s) | Elevation(°) | Azimuth(°)")
        print("-" * 80)

        for distance, wind_speed, wind_dir, density in test_cases:
            result = fitter.calculate_with_formulas(distance, wind_speed, wind_dir, density)
            print(f"{distance:7.0f} | {wind_speed:9.1f} | {wind_dir:8.0f} | {density:4.1f} | "
                  f"{result['velocity']:9.1f} | {result['elevation']:8.1f} | {result['azimuth']:9.1f}")

        # Plot fitting results
        print("\nPlotting fitting results...")
        error_stats = fitter.plot_fitting_results(df, fit_results)

        print("\nError Statistics:")
        print(f"Velocity: mean={error_stats['velocity_error_stats']['mean']:.2f} m/s, "
              f"std={error_stats['velocity_error_stats']['std']:.2f} m/s")
        print(f"Elevation: mean={error_stats['elevation_error_stats']['mean']:.2f}°, "
              f"std={error_stats['elevation_error_stats']['std']:.2f}°")
        print(f"Azimuth: mean={error_stats['azimuth_error_stats']['mean']:.2f}°, "
              f"std={error_stats['azimuth_error_stats']['std']:.2f}°")

        # Produce final manual calculation guide (English)
        print("\n" + "="*60)
        print("FINAL MANUAL CALCULATION GUIDE")
        print("="*60)

        manual_guide = f"""
Empirical artillery manual (derived from fitted data)
===============================================

1. Velocity formula:
   v = {coeffs['velocity_base']:.1f} + {coeffs['velocity_distance_coef']:.4f}×D + {coeffs['velocity_wind_coef']:.4f}×W + {coeffs['velocity_wind_dir_coef']:.4f}×W×cos(phi) + {coeffs['velocity_density_coef']:.4f}/√ρ

2. Elevation formula:
   θ = {coeffs['elevation_base']:.1f} + {coeffs['elevation_distance_coef']:.4f}×D + {coeffs['elevation_wind_coef']:.4f}×W + {coeffs['elevation_wind_dir_coef']:.4f}×W×sin(phi)

3. Azimuth formula:
   α = {coeffs['azimuth_wind_coef']:.4f} × W × sin(phi)

Where:
- D: target distance (m), typical range: 1000-1500 m
- W: wind speed (m/s)
- phi: wind direction (degrees), convention used in features
- ρ: air density (kg/m³), used range: 1.1-1.3

Quick reference:
- Per +100 m distance, velocity changes ≈ {coeffs['velocity_distance_coef']*100:.1f} m/s
- Per +1 m/s headwind, velocity changes ≈ {coeffs['velocity_wind_coef'] + coeffs['velocity_wind_dir_coef']:.3f} m/s
- Density change 1.2→1.1 causes velocity change ≈ {(coeffs['velocity_density_coef']*(1/np.sqrt(1.1)-1/np.sqrt(1.2))):.1f} m/s
        """
        print(manual_guide)

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the CSV files are in the current directory.")

if __name__ == "__main__":
    main()