# Necessary Imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# Libraries for Multicollinearity and Factor Analysis
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
from factor_analyzer import FactorAnalyzer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
# Libraries for Time Series Analysis
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# Libraries for Econometrics Modeling
import statsmodels.api as sm
from linearmodels.panel import PanelOLS


class PanelDataProcessor:
    def __init__(self, wb_path, owid_path):
        """
        Initialize the processor with file paths.
        """
        self.wb_path = wb_path
        self.owid_path = owid_path

        # We will store the dataframes here
        self.wb_data = None
        self.owid_data = None
        self.panel_data = None

    def clean_worldbank(self):
        """
        Load and clean the World Bank dataset.
        """
        print("Cleaning World Bank dataset...")
        df = pd.read_csv(self.wb_path)

        # The World Bank often uses '..' for missing values. Convert these to NaN.
        df.replace('..', np.nan, inplace=True)

        # Rename standard index columns so they match OWID data
        df.rename(columns={
            'Country Name': 'country',
            'Country Code': 'iso_code',
            'Time': 'year'
        }, inplace=True)

        # Drop 'Time Code' as it's redundant
        if 'Time Code' in df.columns:
            df.drop(columns=['Time Code'], inplace=True)

        # Rename the long metric columns to simpler names
        col_mapping = {
            'Access to electricity (% of population) [EG.ELC.ACCS.ZS]': 'access_to_electricity_pct',
            'Access to clean fuels and technologies for cooking (% of population) [EG.CFT.ACCS.ZS]': 'access_to_clean_cooking_pct',
            'Renewable energy consumption (% of total final energy consumption) [EG.FEC.RNEW.ZS]': 'renewable_energy_share_pct',
            'GDP per capita (constant LCU) [NY.GDP.PCAP.KN]': 'gdp_per_capita',
            'GDP growth (annual %) [NY.GDP.MKTP.KD.ZG]': 'gdp_growth_pct',
            'Carbon dioxide (CO2) emissions excluding LULUCF per capita (t CO2e/capita) [EN.GHG.CO2.PC.CE.AR5]': 'co2_emissions_per_capita',
            'Population density (people per sq. km of land area) [EN.POP.DNST]': 'population_density',
            'Land area (sq. km) [AG.LND.TOTL.K2]': 'land_area_sq_km'
        }
        df.rename(columns=col_mapping, inplace=True)

        # Ensure the 'year' column is treated as an integer
        # Drop rows where 'year' might be a footer/text or NaN
        df.dropna(subset=['year'], inplace=True)
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df.dropna(subset=['year'], inplace=True)
        df['year'] = df['year'].astype(int)

        # Convert metric columns to numeric (floats)
        metrics = list(col_mapping.values())
        for col in metrics:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        self.wb_data = df
        print(f"World Bank dataset cleaned. Shape: {self.wb_data.shape}")

    def clean_owid(self):
        """
        Load and clean the Our World in Data dataset.
        """
        print("Cleaning OWID dataset...")
        df = pd.read_csv(self.owid_path)

        # Ensure standard types for merging
        df.dropna(subset=['year', 'iso_code'], inplace=True)
        df['year'] = df['year'].astype(int)

        self.owid_data = df
        print(f"OWID dataset cleaned. Shape: {self.owid_data.shape}")

    def merge_to_panel(self):
        """
        Merge datasets into a final panel format using 'iso_code' and 'year'.
        """
        print("Merging datasets into a panel...")

        if self.wb_data is None or self.owid_data is None:
            raise ValueError("Datasets have not been loaded and cleaned yet. Run cleaning methods first.")

        owid_drop_country = self.owid_data.drop(columns=['country'])

        self.panel_data = pd.merge(
            self.wb_data,
            owid_drop_country,
            on=['iso_code', 'year'],
            how='outer'
        )

        # Sort values chronologically for each country
        self.panel_data.sort_values(by=['country', 'year'], inplace=True)
        self.panel_data.reset_index(drop=True, inplace=True)

        print(f"Panel dataset successfully created! Shape: {self.panel_data.shape}")

    def save_panel_data(self, output_filename='final_panel_dataset.csv'):
        """
        Export the processed panel dataset.
        """
        if self.panel_data is not None:
            self.panel_data.to_csv(output_filename, index=False)
            print(f"Data saved to {output_filename}")
        else:
            print("No panel data to save. Process the data first.")

    def run_pipeline(self, output_filename='final_panel_dataset.csv'):
        """
        Orchestrator function to run all steps.
        """
        self.clean_worldbank()
        self.clean_owid()
        self.merge_to_panel()
        self.save_panel_data(output_filename)
        return self.panel_data


class PanelDataCleaner:
    def __init__(self, input_filepath):
        """
        Initialize by loading the merged panel dataset.
        """
        print(f"Loading data from {input_filepath}...")
        self.df = pd.read_csv(input_filepath)

    def filter_years(self, start_year, end_year):
        """
        Filter the panel dataset to a specific time period.
        """
        print(f"Filtering dataset for years {start_year} to {end_year}...")
        mask = (self.df['year'] >= start_year) & (self.df['year'] <= end_year)
        self.df = self.df[mask].copy()

    def trim_variables(self):
        """
        Keep only the variables of interest.
        """
        print("Trimming variables...")

        variables_to_keep = [
            # Identifiers
            'country', 'iso_code', 'year',

            # Dependent Variable
            'renewable_energy_share_pct',

            # Independent Variables
            'access_to_clean_cooking_pct',
            'land_area_sq_km',
            'gdp_per_capita',
            'fossil_electricity',
            'nuclear_electricity',
            'energy_per_capita',
            'co2_emissions_per_capita',

            # Bonus Controls
            'access_to_electricity_pct',
            'gdp_growth_pct',
            'population_density',
            'low_carbon_electricity',
            'low_carbon_share_energy',
            'greenhouse_gas_emissions',
            'electricity_demand'
        ]

        # Keep only the columns that actually exist in the dataframe to avoid KeyErrors
        available_cols = [col for col in variables_to_keep if col in self.df.columns]
        self.df = self.df[available_cols].copy()

    def impute_missing_values(self):
        """
        Impute missing values per country:
        1. Linear interpolation (median between values) for gaps in the middle.
        2. Forward fill for newer missing years (takes previous valid year).
        3. Backward fill for older missing years (takes future valid year).
        """
        print("Imputing missing values...")

        # Ensure data is sorted properly before filling
        self.df.sort_values(by=['iso_code', 'year'], inplace=True)

        def fill_country_gaps(group):
            # 1. Fill middle gaps mathematically
            group = group.interpolate(method='linear')
            # 2. Fill newest years with the most recent available data
            group = group.ffill()
            # 3. Fill oldest years with the earliest available data
            group = group.bfill()
            return group

        # Identify only numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.drop('year')

        # Apply the filling logic, grouping by country so data doesn't bleed across borders
        self.df[numeric_cols] = self.df.groupby('iso_code', group_keys=False)[numeric_cols].apply(fill_country_gaps)

    def save_data(self, output_filename):
        """
        Export the finalized, cleaned panel dataset.
        """
        self.df.to_csv(output_filename, index=False)
        print(f"Data successfully saved to {output_filename}")
        print(f"Final dataset shape: {self.df.shape}")

    def run_cleaning_pipeline(self, start_year=2012, end_year=2021, output_filename='Final_Cleaned_Panel.csv'):
        """
        Orchestrator function to run all cleaning steps in order.
        """
        self.filter_years(start_year, end_year)
        self.trim_variables()
        self.impute_missing_values()

        # Drop rows where the dependent variable is STILL missing after all imputation
        # (This happens if a country has zero data for renewable energy across all years)
        self.df.dropna(subset=['renewable_energy_share_pct'], inplace=True)

        self.save_data(output_filename)
        return self.df


class DataExploration:
    def __init__(self, dataframe):
        """
        Initialize the DataExploration class with your cleaned dataframe.
        """
        self.df = dataframe.copy()

        # Isolate only the numeric columns for our mathematical checks (ignore year/text)
        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if 'year' in self.numeric_cols:
            self.numeric_cols.remove('year')

        # Drop any leftover NaNs just for the statistical tests
        self.analysis_data = self.df[self.numeric_cols].dropna()

        print(f"Initialized DataExploration with {self.analysis_data.shape[0]} valid records and {len(self.numeric_cols)} numeric variables.")

    def descriptive_statistics(self, output_filename='Descriptive_Statistics.csv'):
        """
        Calculates and exports standard descriptive statistics (mean, min, max, std).
        """
        print("\n--- Calculating Descriptive Statistics ---")
        desc_stats = self.analysis_data.describe().T
        desc_stats.to_csv(output_filename)
        print(f"Saved to {output_filename}")

        # Display a quick preview of Min, Max, and Mean to spot scale differences
        print(desc_stats[['mean', 'min', 'max', 'std']].round(2))
        return desc_stats

    def plot_correlation_heatmap(self, save_path='correlation_heatmap.png'):
        """
        Generates and saves a Pearson Correlation Heatmap.
        """
        print("\n--- Generating Correlation Heatmap ---")
        corr_matrix = self.analysis_data.corr()

        plt.figure(figsize=(14, 12))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, 
                    cbar_kws={'label': 'Pearson Correlation'})
        plt.title("Correlation Matrix of Energy & Economic Indicators", fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        plt.savefig(save_path)
        plt.show()
        print(f"Heatmap saved to {save_path}")

    def check_multicollinearity(self, output_filename='VIF_Results.csv'):
        """
        Calculates the Variance Inflation Factor (VIF) to detect multicollinearity.
        A VIF > 10 indicates problematic collinearity.
        """
        print("\n--- Checking Multicollinearity (VIF) ---")

        vif_data = pd.DataFrame()
        vif_data["Variable"] = self.analysis_data.columns

        # Calculate VIF for each feature
        vif_data["VIF"] = [
            variance_inflation_factor(self.analysis_data.values, i) 
            for i in range(self.analysis_data.shape[1])
        ]

        # Sort by highest VIF
        vif_data.sort_values(by="VIF", ascending=False, inplace=True)
        vif_data.to_csv(output_filename, index=False)

        print("Variables with VIF > 10 (Consider dropping/transforming these):")
        print(vif_data[vif_data["VIF"] > 10].to_string(index=False))
        return vif_data

    def factor_analysis_prerequisites(self):
        """
        Runs Bartlett's Test of Sphericity and the KMO Test.
        Data MUST be standardized before calculating KMO properly.
        """
        print("\n--- Checking Factor Analysis Prerequisites ---")

        # 1. Standardize the data (mean=0, variance=1)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.analysis_data)
        scaled_df = pd.DataFrame(scaled_data, columns=self.analysis_data.columns)

        # 2. Bartlett’s Test of Sphericity
        # Tests the hypothesis that your correlation matrix is an identity matrix (variables are unrelated)
        # We want a p-value < 0.05
        chi_square_value, p_value = calculate_bartlett_sphericity(scaled_df)
        print(f"Bartlett's Test P-value: {p_value}")
        if p_value < 0.05:
            print("  -> SUCCESS: Variables are correlated enough for Factor Analysis.")
        else:
            print("  -> WARNING: Variables are not correlated enough for Factor Analysis.")

        # 3. Kaiser-Meyer-Olkin (KMO) Test
        # Measures the proportion of variance among variables that might be common variance.
        # We want an overall KMO > 0.60
        kmo_all, kmo_model = calculate_kmo(scaled_df)
        print(f"\nOverall KMO Score: {kmo_model:.3f}")

        if kmo_model >= 0.60:
            print("  -> SUCCESS: KMO score is adequate for Factor Analysis.")
        else:
            print("  -> WARNING: KMO score is too low. You may need to drop variables with low individual KMOs.")

        # Show individual KMOs to identify weak variables
        kmo_vars = pd.DataFrame({'Variable': self.analysis_data.columns, 'KMO': kmo_all})
        kmo_vars.sort_values(by='KMO', ascending=True, inplace=True)
        print("\nVariables with the lowest KMO scores (< 0.50 should be dropped):")
        print(kmo_vars[kmo_vars['KMO'] < 0.60].to_string(index=False))

    def run_all(self):
        """
        Executes all exploration functions.
        """
        self.descriptive_statistics()
        self.plot_correlation_heatmap()
        self.check_multicollinearity()
        self.factor_analysis_prerequisites()


class FactorAnalysisPrep:
    def __init__(self, dataframe, target_variable):
        """
        Initializes the class by separating dependent and strictly curated independent variables.
        """
        self.df = dataframe.copy().dropna()
        self.y = self.df[target_variable]

        # STRICT MANUAL SELECTION
        selected_features = [
            'access_to_electricity_pct',
            'access_to_clean_cooking_pct',
            'gdp_per_capita',
            'gdp_growth_pct',
            'energy_per_capita',
            'low_carbon_share_energy'
        ]

        # Keep only the ones that exist in the dataframe to avoid key errors
        available_features = [f for f in selected_features if f in self.df.columns]
        self.X = self.df[available_features].copy()

        print(f"Initialized with {len(available_features)} highly curated features.")

    def handle_outliers(self):
        """Caps extreme outliers at 1st and 99th percentiles."""
        print("\n--- Handling Outliers ---")
        for col in self.X.columns:
            lower = self.X[col].quantile(0.01)
            upper = self.X[col].quantile(0.99)
            self.X[col] = self.X[col].clip(lower=lower, upper=upper)
        print("Outliers capped successfully.")

    def handle_skewed_distributions(self):
        """Applies log1p transformation to variables with high skewness (>1.5)."""
        print("\n--- Handling Skewed Distributions ---")
        skewness = self.X.skew()
        skewed_cols = skewness[abs(skewness) > 1.5].index.tolist()

        for col in skewed_cols:
            min_val = self.X[col].min()
            if min_val < 0:
                self.X[col] = np.log1p(self.X[col] - min_val)
            else:
                self.X[col] = np.log1p(self.X[col])
        print(f"Log transformed skewed variables: {skewed_cols}")

    def standardize_data(self):
        """Scales the data to Mean = 0, Standard Deviation = 1."""
        print("\n--- Standardizing Variables ---")
        scaler = StandardScaler()
        self.X_scaled = pd.DataFrame(
            scaler.fit_transform(self.X), 
            columns=self.X.columns, 
            index=self.X.index
        )
        print("Data standardization complete.")

    def run_tests(self):
        """Runs VIF, Bartlett's, and KMO tests to confirm readiness."""
        print("\n--- Final VIF Check ---")
        vif_data = pd.DataFrame()
        vif_data["Variable"] = self.X_scaled.columns
        vif_data["VIF"] = [variance_inflation_factor(self.X_scaled.values, i) for i in range(self.X_scaled.shape[1])]
        print(vif_data.sort_values("VIF", ascending=False).to_string(index=False))

        print("\n--- Factor Analysis Prerequisites ---")
        _, p_value = calculate_bartlett_sphericity(self.X_scaled)
        print(f"Bartlett's Test P-value: {p_value}")
        if p_value < 0.05:
            print("  -> SUCCESS: Variables are correlated enough for Factor Analysis.")
        else:
            print("  -> WARNING: Variables are not correlated enough.")

        kmo_all, kmo_model = calculate_kmo(self.X_scaled)
        print(f"Overall KMO Score: {kmo_model:.3f}")
        if kmo_model >= 0.60:
            print("  -> SUCCESS: KMO score is adequate for Factor Analysis.")
        else:
            print("  -> WARNING: KMO score is below 0.60. Factor Analysis might not yield distinct groups.")

        kmo_df = pd.DataFrame({'Variable': self.X_scaled.columns, 'Individual_KMO': kmo_all})
        print("\nIndividual KMO Scores (>0.5 is good):")
        print(kmo_df.sort_values('Individual_KMO').to_string(index=False))

    def run_pipeline(self):
        """
        Orchestrator method to run all preparation steps with a single call.
        Returns the scaled features (X) and the dependent variable (y).
        """
        self.handle_outliers()
        self.handle_skewed_distributions()
        self.standardize_data()
        self.run_tests()

        print("\nPipeline Complete! Data is ready for Factor Analysis.")
        return self.X_scaled, self.y


class CountryProfiler:
    def __init__(self, X_scaled, original_df):
        """
        Takes the scaled independent variables for math.
        Filters the original dataframe to ONLY include the rows that survived 
        the missing-value drop in the previous step to avoid length mismatches.
        """
        self.X = X_scaled.copy()

        # FIX: Filter the original dataframe using the index of the scaled data
        self.df = original_df.loc[self.X.index].copy()

        self.fa_scores = None

    def perform_factor_analysis(self):
        """
        Step 1: Uses the Kaiser Criterion (Eigenvalues > 1) to determine the optimal 
        number of factors, then extracts the factors using Varimax rotation.
        """
        print("\n--- Step 1: Factor Analysis ---")

        # Initial fit to get eigenvalues
        fa_temp = FactorAnalyzer(rotation=None)
        fa_temp.fit(self.X)
        eigenvalues, _ = fa_temp.get_eigenvalues()

        # Kaiser Criterion: Keep factors with eigenvalues > 1
        n_factors = sum(eigenvalues > 1)
        print(f"Optimal number of latent factors found (Eigenvalue > 1): {n_factors}")

        # Fit the final Factor Model with Varimax rotation (makes factors easier to interpret)
        fa_final = FactorAnalyzer(n_factors=n_factors, rotation='varimax')
        fa_final.fit(self.X)

        # Transform the original data into the new Factor Scores
        self.fa_scores = fa_final.transform(self.X)

        # Print Factor Loadings to see what each factor represents
        loadings = pd.DataFrame(fa_final.loadings_, index=self.X.columns, 
                                columns=[f"Factor_{i+1}" for i in range(n_factors)])
        print("\nFactor Loadings (What each factor represents):")
        print(loadings.round(2))

        return self.fa_scores

    def determine_optimal_clusters(self, max_k=8):
        """
        Step 2: Uses the Elbow Method and Silhouette Score to find the best number of clusters.
        """
        print("\n--- Step 2: Determining Optimal Clusters ---")
        inertias = []
        silhouettes = []

        K_range = range(2, max_k + 1)
        for k in K_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(self.fa_scores)

            inertias.append(km.inertia_)
            silhouettes.append(silhouette_score(self.fa_scores, labels))

        # Plotting the Elbow and Silhouette curves
        fig, ax1 = plt.subplots(figsize=(8, 5))

        color = 'tab:red'
        ax1.set_xlabel('Number of Clusters (K)')
        ax1.set_ylabel('Inertia (Elbow Method)', color=color)
        ax1.plot(K_range, inertias, marker='o', color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  
        color = 'tab:blue'
        ax2.set_ylabel('Silhouette Score (Higher is better)', color=color)
        ax2.plot(K_range, silhouettes, marker='s', color=color, linestyle='dashed')
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title("Cluster Optimization: Elbow & Silhouette")
        plt.tight_layout()
        plt.savefig("Cluster_Optimization.png")
        print("Cluster optimization plot saved as 'Cluster_Optimization.png'.")

        # Automatically recommend best K based on max silhouette
        best_k = K_range[np.argmax(silhouettes)]
        print(f"Recommended number of clusters (Max Silhouette Score): {best_k}")
        return best_k

    def fit_clusters(self, n_clusters):
        """
        Step 3: Fits K-Means on the Factor Scores and assigns profiles to the original dataframe.
        """
        print(f"\n--- Step 3: Fitting K-Means (K={n_clusters}) ---")
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

        # Predict clusters and attach to the strictly aligned dataframe
        self.df['Profile_Cluster'] = km.fit_predict(self.fa_scores)

        # Print a quick summary of what each profile looks like (using median values to ignore outliers)
        # We need to explicitly check against columns that still exist to avoid errors
        cols_to_summarize = [col for col in self.X.columns if col in self.df.columns]
        profile_summary = self.df.groupby('Profile_Cluster')[cols_to_summarize].median()

        print("\nMedian Characteristics of Each Profile (Unscaled Values):")
        print(profile_summary.round(2))

        return self.df

    def run_profiling(self):
        """
        Runs the full profiling algorithm.
        """
        self.perform_factor_analysis()
        best_k = self.determine_optimal_clusters()
        final_df = self.fit_clusters(n_clusters=best_k)

        print("\nProfiling Complete! Final dataframe is ready for Time Series / FE modeling.")
        return final_df


class TimeSeriesAnalyzer:
    def __init__(self, dataframe, target_var='renewable_energy_share_pct', time_var='year', cluster_var='Profile_Cluster'):
        """
        Initializes the Time Series Analyzer.
        """
        self.df = dataframe.copy()
        self.target = target_var
        self.time = time_var
        self.cluster = cluster_var

        # Create a Global Time Series (Average of all countries per year)
        self.global_ts = self.df.groupby(self.time)[self.target].mean()

        # Create a Cluster-based Time Series (Average per cluster per year)
        if self.cluster in self.df.columns:
            self.cluster_ts = self.df.groupby([self.time, self.cluster])[self.target].mean().unstack()
        else:
            self.cluster_ts = None

        print(f"Initialized Time Series Analyzer for '{self.target}' ({self.df[self.time].min()} - {self.df[self.time].max()})")

    def plot_macro_trends(self):
        """
        1. Macro Trend Analysis
        Purpose: Visualizes the overall trajectory of renewable adoption. 
        Insight: Are we actually transitioning globally? Are the profiles moving at the same speed?
        """
        print("\n--- 1. Macro Trend Analysis ---")
        plt.figure(figsize=(10, 6))

        # Plot Global Trend
        plt.plot(self.global_ts.index, self.global_ts.values, label='Global Average',
                 linewidth=3, color='black', linestyle='--')

        # Plot Cluster Trends
        if self.cluster_ts is not None:
            for cluster in self.cluster_ts.columns:
                plt.plot(self.cluster_ts.index, self.cluster_ts[cluster], 
                         label=f'Profile Cluster {cluster}', linewidth=2)

        plt.title(f"Macro Trend: {self.target} Over Time", fontsize=14)
        plt.xlabel("Year")
        plt.ylabel(f"Average {self.target}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("TSA_Macro_Trends.png")
        plt.show()
        print("Saved 'TSA_Macro_Trends.png'.")

    def test_stationarity(self):
        """
        2. Augmented Dickey-Fuller (ADF) Test
        Purpose: Checks if the data is "Stationary" (has a constant mean and variance over time).
        Insight: If data is NON-stationary (trending endlessly), running standard Fixed Effects
        regressions can lead to "Spurious Regressions" (fake statistical significance). You might
        need to use the First-Difference of the variable in your final model.
        """
        print("\n--- 2. Stationarity Test (Augmented Dickey-Fuller) ---")

        # Run ADF on the global average
        result = adfuller(self.global_ts.dropna())

        print(f"ADF Statistic: {result[0]:.4f}")
        print(f"P-value: {result[1]:.4f}")

        if result[1] < 0.05:
            print("Conclusion: The series is STATIONARY (Reject the null hypothesis).")
            print("Econometric Impact: Safe to use standard Fixed/Random Effects modeling.")
        else:
            print("Conclusion: The series is NON-STATIONARY (Fail to reject null hypothesis).")
            print("Econometric Impact: The variable has a time-trend. You should include 'Year' fixed effects in your final model, or use the Year-over-Year change (first difference).")

    def plot_autocorrelation(self, lags=4):
        """
        3. Autocorrelation (ACF) & Partial Autocorrelation (PACF)
        Purpose: Measures how much "inertia" the variable has. Does last year's adoption rate perfectly predict this year's?
        Insight: If autocorrelation is extremely high, your model will suffer from Serial Correlation.
        You may need a "Dynamic Panel Model" (like Arellano-Bond) or to include a lagged dependent variable (t-1).
        """
        print("\n--- 3. Autocorrelation Analysis (Inertia) ---")

        # We use a small number of lags because panel timeframes are usually short (e.g., 10 years)
        max_lags = min(lags, len(self.global_ts) - 2)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # ACF Plot
        plot_acf(self.global_ts.dropna(), ax=axes[0], lags=max_lags, title="Autocorrelation (ACF)")
        axes[0].set_xlabel("Lags (Years)")
        axes[0].set_ylabel("Correlation")

        # PACF Plot
        plot_pacf(self.global_ts.dropna(), ax=axes[1], lags=max_lags, title="Partial Autocorrelation (PACF)")
        axes[1].set_xlabel("Lags (Years)")
        axes[1].set_ylabel("Correlation")

        plt.tight_layout()
        plt.savefig("TSA_Autocorrelation.png")
        plt.show()

        print(f"Saved 'TSA_Autocorrelation.png'.")
        print("Interpretation: If the bar at Lag 1 is very tall (close to 1.0), renewable energy adoption is highly path-dependent.")

    def analyze_convergence(self):
        """
        4. Convergence / Divergence Analysis (Variance over time)
        Purpose: Calculates the Standard Deviation of countries for each year.
        Insight: Are countries adopting similar renewable shares over time (Convergence / SD going down)? 
        Or is the gap between the Green Leaders and Fossil Reliants widening (Divergence / SD going up)?
        """
        print("\n--- 4. Policy Convergence Analysis ---")

        # Calculate standard deviation across all countries for each year
        variance_ts = self.df.groupby(self.time)[self.target].std()

        plt.figure(figsize=(10, 5))
        plt.plot(variance_ts.index, variance_ts.values, color='purple', marker='o', linewidth=2)
        plt.title("Convergence Check: Standard Deviation of Renewable Adoption Over Time", fontsize=14)
        plt.xlabel("Year")
        plt.ylabel(f"Standard Deviation of {self.target}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("TSA_Convergence.png")
        plt.show()

        print("Saved 'TSA_Convergence.png'.")

        # Simple trend check
        if variance_ts.iloc[-1] < variance_ts.iloc[0]:
            print("Conclusion: CONVERGENCE detected. The gap between countries is shrinking.")
        else:
            print("Conclusion: DIVERGENCE detected. The gap between leading and lagging countries is widening.")

    def run_all(self):
        """
        Executes all time series analyses.
        """
        self.plot_macro_trends()
        self.test_stationarity()
        self.plot_autocorrelation()
        self.analyze_convergence()
        print("\nTime Series Analysis Complete!")


class AdvancedDataProcessing:
    def __init__(self, dataframe, target_variable):
        """
        Takes the Final Panel (which contains your 18 cleaned variables + Profile Cluster).
        Automatically detects the independent variables.
        """
        self.df = dataframe.copy()
        self.target_variable = target_variable

        # Automatically detect all numeric columns
        all_numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()

        # Exclude structural columns from independent variable processing
        exclude_cols = ['year', 'Profile_Cluster', self.target_variable]

        # Dynamically set the independent variables to process
        self.raw_independent_vars = [col for col in all_numeric_cols if col not in exclude_cols]

        print(f"Initialized AdvancedDataProcessing.")
        print(f"Dynamically detected {len(self.raw_independent_vars)} independent variables from the cleaned panel.")

    def handle_outliers(self):
        """Caps extreme outliers at the 1st and 99th percentiles."""
        print("\n--- Handling Outliers (Winsorizing at 1% and 99%) ---")
        for col in self.raw_independent_vars:
            lower = self.df[col].quantile(0.01)
            upper = self.df[col].quantile(0.99)
            self.df[col] = self.df[col].clip(lower=lower, upper=upper)

    def handle_skewed_distributions(self):
        """Applies a logarithmic transformation to heavily skewed variables."""
        print("\n--- Applying Log Transformations to Skewed Variables ---")
        skewness = self.df[self.raw_independent_vars].skew()
        skewed_cols = skewness[abs(skewness) > 1.5].index.tolist()

        for col in skewed_cols:
            min_val = self.df[col].min()
            if min_val < 0:
                self.df[col] = np.log1p(self.df[col] - min_val)
            else:
                self.df[col] = np.log1p(self.df[col])

        print(f"Log transformed variables: {skewed_cols}")

    def drop_highly_correlated(self, threshold=0.80):
        """Drops variables with a correlation > threshold."""
        print(f"\n--- Dropping Highly Correlated Variables (> {threshold}) ---")

        X = self.df[self.raw_independent_vars]
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        print(f"Variables dropped due to multicollinearity: {to_drop}")

        self.surviving_vars = [var for var in self.raw_independent_vars if var not in to_drop]
        print(f"Surviving independent variables for regression: {len(self.surviving_vars)}")

    def run_pipeline(self):
        self.handle_outliers()
        self.handle_skewed_distributions()
        self.drop_highly_correlated()

        print("\nData Processing Complete! Ready for Econometrics.")
        return self.df, self.surviving_vars


class AdvancedPanelModeler:
    def __init__(self, dataframe, independent_vars, target_var='renewable_energy_share_pct', cluster_var='Profile_Cluster'):
        """
        Takes the transformed dataframe and dynamic variables directly from Step 7.
        """
        self.df = dataframe.copy()

        # Ensure iso_code and year are set as the MultiIndex (required for linearmodels)
        if 'iso_code' in self.df.columns and 'year' in self.df.columns:
            self.df = self.df.set_index(['iso_code', 'year'])

        self.target = target_var
        self.cluster = cluster_var

        # Keep only the dynamically passed variables that actually exist in the dataframe
        self.independent_vars = [var for var in independent_vars if var in self.df.columns]

        print("\nInitialized AdvancedPanelModeler.")
        print(f"Ready to run econometrics on {len(self.independent_vars)} independent variables.")

    def run_pw2008_cre_fractional_probit(self, cluster_id):
        """
        Papke & Wooldridge (2008): Correlated Random Effects (CRE) Fractional Probit.
        Uses the Mundlak device to control for unobserved heterogeneity.
        """
        print(f"\n{'='*70}")
        print(f" PAPKE & WOOLDRIDGE (2008) CRE FRACTIONAL PROBIT: CLUSTER {cluster_id}")
        print(f"{'='*70}")

        cluster_data = self.df[self.df[self.cluster] == cluster_id].copy()

        # We need the index 'iso_code' available as a column for grouping
        reg_data = cluster_data[[self.target] + self.independent_vars].dropna().reset_index()

        # 1. Convert Target to Fraction (0.0 to 1.0)
        y_frac = reg_data[self.target] / 100.0
        X = reg_data[self.independent_vars]

        # 2. Chamberlain-Mundlak Device: Calculate country-level means for all X variables
        X_means = X.groupby(reg_data['iso_code']).transform('mean')
        X_means.columns = [f"{col}_mean" for col in X_means.columns]

        # Combine original X, the Mundlak means, and a constant
        X_combined = pd.concat([X, X_means], axis=1)
        X_combined = sm.add_constant(X_combined)

        try:
            # Fit GLM with Binomial family and Probit link
            glm_model = sm.GLM(y_frac, X_combined, family=sm.families.Binomial(link=sm.families.links.Probit()))

            # Cluster standard errors by country (iso_code)
            results = glm_model.fit(cov_type='cluster', cov_kwds={'groups': reg_data['iso_code']})
            # print(results.summary()) # Uncomment if you want the raw statsmodels output too
            return results
        except Exception as e:
            print(f"PW2008 Model failed: {e}")
            return None

    def run_ramalho2017_transformed_fe(self, cluster_id):
        """
        Ramalho & Ramalho (2017): Log-Odds Transformation for Panel Fixed Effects.
        """
        print(f"\n{'='*70}")
        print(f" RAMALHO (2017) TRANSFORMED FIXED EFFECTS: CLUSTER {cluster_id}")
        print(f"{'='*70}")

        cluster_data = self.df[self.df[self.cluster] == cluster_id].copy()
        reg_data = cluster_data[[self.target] + self.independent_vars].dropna()

        # 1. Convert Target to Fraction
        y_frac = reg_data[self.target] / 100.0

        # 2. Boundary Adjustment (Log-odds cannot mathematically handle exact 0 or 1)
        y_frac_adj = np.clip(y_frac, 0.001, 0.999)

        # 3. Apply the Log-Odds Transformation
        y_trans = np.log(y_frac_adj / (1 - y_frac_adj))

        X = reg_data[self.independent_vars]
        X = sm.add_constant(X)

        try:
            # Run Panel Fixed Effects
            model = PanelOLS(y_trans, X, entity_effects=True, time_effects=True)
            results = model.fit(cov_type='robust')
            # print(results.summary) # Uncomment if you want the raw linearmodels output too
            return results
        except Exception as e:
            print(f"Ramalho2017 Model failed: {e}")
            return None


def generate_comparison_table(pw_results, ramalho_results, cluster_id):
    """Helper function to print a clean, publication-style comparison table."""
    print(f"\n\n{'*'*80}")
    print(f" FINAL ECONOMETRIC COMPARISON: PROFILE CLUSTER {cluster_id}")
    print(f" Dependent Variable: Renewable Energy Share (%)")
    print(f"{'*'*80}")

    pw_df, ram_df = pd.DataFrame(), pd.DataFrame()

    if pw_results is not None:
        pw_df['PW_Coef'] = pw_results.params
        pw_df['PW_Pval'] = pw_results.pvalues

    if ramalho_results is not None:
        ram_df['Ram_Coef'] = ramalho_results.params
        ram_df['Ram_Pval'] = ramalho_results.pvalues

    comparison = pd.concat([pw_df, ram_df], axis=1)

    def format_output(row, prefix):
        coef = row.get(f'{prefix}_Coef')
        pval = row.get(f'{prefix}_Pval')
        if pd.isna(coef) or pd.isna(pval): return "-"
        stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else ""
        return f"{coef:>8.4f} {stars:<3}"

    comparison['PW2008 (CRE Probit)'] = comparison.apply(lambda r: format_output(r, 'PW'), axis=1)
    comparison['Ramalho2017 (Log-Odds FE)'] = comparison.apply(lambda r: format_output(r, 'Ram'), axis=1)

    final_table = comparison[['PW2008 (CRE Probit)', 'Ramalho2017 (Log-Odds FE)']]

    # Push Mundlak means to the bottom
    main_vars = [idx for idx in final_table.index if not str(idx).endswith('_mean')]
    mean_vars = [idx for idx in final_table.index if str(idx).endswith('_mean')]
    final_table = final_table.loc[main_vars + mean_vars]

    print(final_table.to_string())
    print("-" * 80)
    print(" Significance levels:  *** p<0.01,  ** p<0.05,  * p<0.10")
    print(" Note: Magnitudes are not directly comparable due to different link functions.")
    return final_table
