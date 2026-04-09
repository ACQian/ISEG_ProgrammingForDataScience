# Necessary imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# Libraries for Multicollinearity and Factor Analysis checks
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
from factor_analyzer import FactorAnalyzer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


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


class AdvancedDataProcessing:
    def __init__(self, dataframe, target_variable):
        """
        Initialize the class with the cleaned panel dataframe.
        Separates the Dependent Variable from the Independent Variables.
        """
        self.df = dataframe.copy()
        self.target_variable = target_variable

        # Drop rows with NaN in independent vars so our math doesn't fail
        self.df.dropna(inplace=True)

        # Separate X (Independent) and Y (Dependent)
        self.y = self.df[self.target_variable]

        # Select only numeric independent columns (dropping identifiers like Country/Year)
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if 'year' in numeric_cols:
            numeric_cols.remove('year')
        if target_variable in numeric_cols:
            numeric_cols.remove(target_variable)

        self.X = self.df[numeric_cols]
        print(f"Initialized with {self.X.shape[1]} independent variables.")

    def handle_outliers(self, lower_percentile=0.01, upper_percentile=0.99):
        """
        Applies Winsorization: Caps extreme outliers at the 1st and 99th percentiles.
        This prevents massive countries (like Russia) from dominating the variance.
        """
        print(f"\n--- Handling Outliers (Capping at {lower_percentile*100}th and {upper_percentile*100}th percentiles) ---")

        for col in self.X.columns:
            lower_bound = self.X[col].quantile(lower_percentile)
            upper_bound = self.X[col].quantile(upper_percentile)

            # Clip the values so anything below lower_bound becomes lower_bound, etc.
            self.X[col] = self.X[col].clip(lower=lower_bound, upper=upper_bound)

        print("Outliers capped successfully.")

    def handle_skewed_distributions(self, skew_threshold=1.5):
        """
        Detects highly skewed variables and applies a logarithmic transformation.
        Adjusts for negative values (like GDP growth) before applying the log.
        """
        print(f"\n--- Handling Skewed Distributions (Threshold: |Skew| > {skew_threshold}) ---")

        skewness = self.X.skew()
        highly_skewed_cols = skewness[abs(skewness) > skew_threshold].index.tolist()

        print(f"Highly skewed variables detected: {highly_skewed_cols}")

        for col in highly_skewed_cols:
            min_val = self.X[col].min()

            # If the variable has negative values or zeros, we shift it to be > 0 before taking the log
            # np.log1p computes log(1 + x) which is great for variables with zeros
            if min_val < 0:
                self.X[col] = np.log1p(self.X[col] - min_val)
            else:
                self.X[col] = np.log1p(self.X[col])

        print("Log transformations applied to skewed variables.")

    def drop_highly_correlated(self, threshold=0.80):
        """
        Identifies and drops variables that have a Pearson correlation higher than the threshold.
        """
        print(f"\n--- Dropping Highly Correlated Variables (Threshold: {threshold}) ---")

        corr_matrix = self.X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

        print(f"Variables dropped due to correlation > {threshold}: {to_drop}")

        self.X = self.X.drop(columns=to_drop)
        print(f"Remaining variables: {self.X.columns.tolist()}")

    def standardize_data(self):
        """
        Scales the independent variables using StandardScaler (mean=0, variance=1).
        """
        print("\n--- Standardizing (Scaling) Variables ---")
        scaler = StandardScaler()

        scaled_array = scaler.fit_transform(self.X)
        self.X_scaled = pd.DataFrame(scaled_array, columns=self.X.columns, index=self.X.index)

        print("Data successfully scaled (Mean ~ 0, Standard Deviation = 1).")

    def plot_updated_heatmap(self, output_file='reduced_correlation_heatmap.png'):
        """
        Plots the correlation heatmap for the scaled, reduced variables.
        """
        print("\n--- Plotting Updated Heatmap ---")
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.X_scaled.corr(), annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
        plt.title("Correlation Matrix After Outlier/Skew Fixes & Dropping Vars", fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_file)
        # plt.show() # Uncomment if running in a notebook cell to view inline
        print(f"Heatmap saved as '{output_file}'.")

    def recheck_vif(self):
        """
        Recalculates the Variance Inflation Factor to ensure multicollinearity is resolved.
        """
        print("\n--- Re-checking Multicollinearity (VIF) ---")
        vif_data = pd.DataFrame()
        vif_data["Variable"] = self.X_scaled.columns
        vif_data["VIF"] = [
            variance_inflation_factor(self.X_scaled.values, i) 
            for i in range(self.X_scaled.shape[1])
        ]
        vif_data.sort_values(by="VIF", ascending=False, inplace=True)
        print(vif_data.to_string(index=False))

    def test_factor_analysis_readiness(self):
        """
        Runs Bartlett's and KMO tests on the clean, scaled independent dataset.
        """
        print("\n--- Testing Factor Analysis Readiness ---")

        # Bartlett’s Test
        chi_square_value, p_value = calculate_bartlett_sphericity(self.X_scaled)
        print(f"Bartlett's Test P-value: {p_value}")
        if p_value < 0.05:
            print("  -> SUCCESS: Variables are correlated enough for Factor Analysis.")
        else:
            print("  -> WARNING: Variables are not correlated enough.")

        # KMO Test
        kmo_all, kmo_model = calculate_kmo(self.X_scaled)
        print(f"Overall KMO Score: {kmo_model:.3f}")
        if kmo_model >= 0.60:
            print("  -> SUCCESS: KMO score is adequate for Factor Analysis.")
        else:
            print("  -> WARNING: KMO score is below 0.60. Factor Analysis might not yield distinct groups.")

    def run_pipeline(self):
        """
        Runs the full processing pipeline in logical statistical order.
        """
        # 1. Cap outliers first so they don't distort the skewness test
        self.handle_outliers()

        # 2. Fix skewness so correlations aren't dominated by long tails
        self.handle_skewed_distributions()

        # 3. Drop highly correlated variables
        self.drop_highly_correlated(threshold=0.80)

        # 4. Standardize the fixed data
        self.standardize_data()

        # 5. Review results
        self.plot_updated_heatmap()
        self.recheck_vif()
        self.test_factor_analysis_readiness()

        # Return the final scaled X dataset, and the separate y variable for modeling
        return self.X_scaled, self.y


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
        Takes the scaled independent variables for math, 
        and the original dataframe so we can attach the cluster labels back to it.
        """
        self.X = X_scaled.copy()
        self.df = original_df.copy()
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

        # Predict clusters and attach to original dataframe
        self.df['Profile_Cluster'] = km.fit_predict(self.fa_scores)

        # Print a quick summary of what each profile looks like (using median values to ignore outliers)
        profile_summary = self.df.groupby('Profile_Cluster')[self.X.columns].median()
        print("\nMedian Characteristics of Each Profile:")
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

# ==========================================
# How to run the class
# ==========================================
if __name__ == "__main__":
    # Ensure X_for_FA and df_clean are loaded from your previous Data Prep script
    # df_clean = pd.read_csv('Cleaned_Analysis_Ready_Panel.csv')
    
    profiler = CountryProfiler(X_scaled=X_for_FA, original_df=df_clean)
    
    # Run the pipeline
    final_panel_with_profiles = profiler.run_profiling()
    
    # Save the finalized dataset!
    final_panel_with_profiles.to_csv("Final_Panel_With_Profiles.csv", index=False)
