import pandas as pd
import numpy as np


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
        df.dropna(subset=['year', 'iso_code'], inplace=True) # ISO code is crucial for a global panel merge
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

        # We perform an outer merge to capture all country-years, 
        # or you can use 'inner' if you only want rows present in BOTH datasets.
        # Here we use 'outer' to retain maximum data

        # Notice we merge on both 'iso_code' and 'year'
        # We also drop the duplicated 'country' column from one of the sets before merging to avoid country_x, country_y
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
