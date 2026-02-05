"""
Lab 02: Game Dynamics Analysis with Python Classes
Course: Programming for Data Science
Professor: Dr. Carlos J. Costa

This script implements:
1. A data loader for reading Excel files from a URL
2. An OLS regression analyzer
3. A correlation analyzer with heatmap visualization
4. Grouped descriptive statistics with mean ± std plots
5. A refactored, integrated, modular design
"""

# =============================================================================
# IMPORTS
# =============================================================================

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
# To handle SSL certificate verification for URL access
import ssl
import urllib.request
import io


# =============================================================================
# TASK 1: FILE DATA READER
# =============================================================================

class GameDataLoader:
    """
    Loads the GameTurns Excel dataset from a URL into a pandas DataFrame.
    Handles errors gracefully if the file cannot be accessed.
    """

    def __init__(self, url: str):
        """
        Attributes:
            url (str): URL pointing to the Excel file.
        """
        self.url = url
        self.df = None

    def load(self):
        """
        Loads the first worksheet of the Excel file,
        bypassing SSL verification if necessary.
        """
        try:
            print(f"Loading data from: {self.url}")

            ssl_context = ssl._create_unverified_context()
            with urllib.request.urlopen(self.url, context=ssl_context) as response:
                data = response.read()

            self.df = pd.read_excel(io.BytesIO(data))
            print("Data loaded successfully.\n")
            return self.df

        except Exception as e:
            print(f"Error loading data: {e}")
            return None


# =============================================================================
# BASE ANALYZER (Added for modular design and code reuse)
# =============================================================================

class BaseAnalyzer:
    """
    Base class for all analytical components.
    Stores a pandas DataFrame for reuse across analyses.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df


# =============================================================================
# TASK 2: LINEAR REGRESSION ANALYZER
# =============================================================================

class LinearRegressionAnalyzer(BaseAnalyzer):
    """
    Performs Ordinary Least Squares (OLS) regression using statsmodels.
    Inherits from BaseAnalyzer to access the DataFrame loaded by GameDataLoader.
    """

    def run_ols(self, dependent_var: str, independent_vars: list):
        """
        Runs an OLS regression.

        Attributes:
            dependent_var (str): Name of the dependent variable.
            independent_vars (list): List of independent variable names.

        Returns:
            statsmodels summary object
        """
        y = self.df[dependent_var]
        X = self.df[independent_vars]
        X = sm.add_constant(X)  # add intercept

        model = sm.OLS(y, X).fit()
        return model.summary()


# =============================================================================
# TASK 3: CORRELATION ANALYZER
# =============================================================================

class CorrelationAnalyzer(BaseAnalyzer):
    """
    Computes and visualizes the correlation matrix for numerical variables.
    Inherits from BaseAnalyzer to access the DataFrame loaded by GameDataLoader.
    """

    def compute_matrix(self):
        """
        Computes the correlation matrix.

        Returns:
            Correlation matrix as a pandas DataFrame.
        """
        return self.df.corr(numeric_only=True)

    def plot_heatmap(self):
        """
        Plots the correlation matrix as a heatmap.
        """
        corr = self.compute_matrix()

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Matrix Heatmap")
        plt.show()


# =============================================================================
# TASK 4: GROUPED DESCRIPTIVE STATISTICS
# =============================================================================

class GroupStatisticsAnalyzer(BaseAnalyzer):
    """
    Computes and visualizes descriptive statistics grouped by a column.
    Inherits from BaseAnalyzer to access the DataFrame loaded by GameDataLoader.
    """

    def compute_stats(self, group_col: str):
        """
        Computes mean and standard deviation for numeric columns only,
        grouped by the specified column.

        Attributes:
            group_col (str): Column name to group by.

        Returns:
            DataFrame with mean and std for each numeric variable.
        """
        numeric_df = self.df.select_dtypes(include="number")

        grouped = numeric_df.groupby(self.df[group_col])
        return grouped.agg(['mean', 'std'])

    def plot_mean_std(self, group_col: str, variable: str):
        """
        Plots mean with ±1 standard deviation shading.

        Attributes:
            group_col (str): Grouping column.
            variable (str): Variable to visualize.

        Returns:
            Plot display.
        """
        stats = self.compute_stats(group_col)

        mean = stats[variable]['mean']
        std = stats[variable]['std']
        x = mean.index

        plt.figure(figsize=(10, 6))
        plt.plot(x, mean, label="Mean", marker="o")
        plt.fill_between(
            x,
            mean - std,
            mean + std,
            alpha=0.3,
            label="±1 Std Dev"
        )

        plt.title(f"{variable} by {group_col}")
        plt.xlabel(group_col)
        plt.ylabel(variable)
        plt.legend()
        plt.grid(alpha=0.6)
        plt.show()


if __name__ == "__main__":

    DATA_URL = (
        "https://github.com/masterfloss/FakeNewsData/raw/refs/heads/main/GameTurns.xlsx"
    )

    # -----------------------------
    # Load Data
    # -----------------------------
    loader = GameDataLoader(DATA_URL)
    df = loader.load()

    if df is None:
        print("Program terminated due to data loading failure.")
        exit()

    print(df.head(), "\n")

    # -----------------------------
    # Linear Regression Analysis
    # -----------------------------
    reg = LinearRegressionAnalyzer(df)
    print("OLS REGRESSION RESULTS:\n")
    print(
        reg.run_ols(
            dependent_var="current_followers",
            independent_vars=[
                "round_number",
                "credibility_change",
                "followers_change"
            ]
        )
    )

    # -----------------------------
    # Correlation Analysis
    # -----------------------------
    corr = CorrelationAnalyzer(df)
    corr.plot_heatmap()

    # -----------------------------
    # Grouped Descriptive Statistics
    # -----------------------------
    stats = GroupStatisticsAnalyzer(df)
    grouped_df = stats.compute_stats("round_number")

    print("\nGrouped Descriptive Statistics:\n")
    print(grouped_df)

    stats.plot_mean_std(
        group_col="round_number",
        variable="current_followers"
    )
