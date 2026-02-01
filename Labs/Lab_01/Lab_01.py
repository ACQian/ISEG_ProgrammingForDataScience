# Lab_01: Travel Analytics Lab: Modeling Trips and Places

import pandas as pd

# ==============================================================================
# SECTION 1: CORE LAB (Original Requirements)
# ==============================================================================


class Place:
    """
    This class represents a place with a name and country.
    Attributes:
        city (str): The name of the city or location.
        country (str): The country where the place is located.
    """
    def __init__(self, city, country):
        self.city = city
        self.country = country

    def __str__(self):
        return f"{self.city} ({self.country})"


class Trip:
    """
    This class represents a trip between two places.
    And also includes a method to compute cost per day.
    Attributes:
        origin (Place): The starting point of the trip.
        destination (Place): The endpoint of the trip.
        transport_mode (str): The mode of transportation used for the trip.
        total_cost (float): The total cost of the trip.
        duration_days (int): The duration of the trip in days.
    """
    def __init__(
            self, origin, destination,  # taken from Place class
            transport_mode, total_cost, duration_days  # new attributes
            ):
        self.origin = origin
        self.destination = destination
        self.transport_mode = transport_mode
        self.total_cost = total_cost
        self.duration_days = duration_days

    def cost_per_day(self):
        """
        Calculate the cost per day of the trip.
        Returns:
            float: The cost per day.
        """
        if self.duration_days == 0:
            return 0  # Avoid division by zero
        return self.total_cost / self.duration_days


# ==============================================================================
# SECTION 2: EXTENDED CLASSES (Advanced Features)
# ==============================================================================


class TransportMode:
    """
    Represents a specific mode of transport with emissions and speed data.
    Attributes:
        name (str): Name of the transport mode (e.g., "Plane", "Train").
        co2_per_km (float): CO2 emissions per kilometer (kg/km).
        avg_speed_kmh (float): Average speed in kilometers per hour (km/h).
    """
    def __init__(self, name, co2_per_km, avg_speed_kmh):
        self.name = name
        self.co2_per_km = co2_per_km
        self.avg_speed_kmh = avg_speed_kmh

    def __str__(self):
        return self.name


class EcoTrip(Trip):
    """
    Advanced version of Trip.
    Adds distance tracking and CO2 calculations.
    Attributes:
        transport_mode (object): TransportMode() object required.
        distance_km (float): The distance of the trip in kilometers.
    """
    def __init__(
            self, origin, destination, transport_mode,
            total_cost, duration_days, distance_km):

        # Initialize the parent Trip class
        super().__init__(
            origin, destination, transport_mode,
            total_cost, duration_days)
        self.distance_km = distance_km

    def calculate_co2(self):
        """Calculates total CO2 emissions (kg) based on distance and mode."""
        return self.distance_km * self.transport_mode.co2_per_km

    def est_travel_time_hours(self):
        """Estimates pure travel time in hours."""
        if self.transport_mode.avg_speed_kmh == 0: 
            return 0
        return self.distance_km / self.transport_mode.avg_speed_kmh


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":

    # ---------------------------------------------------------
    # PART 1: REQUIRED METRICS (STATIC ANALYSIS OF ALL TRIPS)
    # ---------------------------------------------------------
    print("\n" + "#"*60)
    print(" PART 1: REQUIRED METRICS")
    print("#"*60)

    # 1. List of Places
    lisbon = Place("Lisbon", "Portugal")
    paris = Place("Paris", "France")
    madrid = Place("Madrid", "Spain")
    berlin = Place("Berlin", "Germany")
    ny = Place("New York", "USA")

    # 2. List of Trips
    trip1 = Trip(lisbon, paris, "Plane", 1600, 6)
    trip2 = Trip(paris, madrid, "Train", 1800, 8)
    trip3 = Trip(lisbon, madrid, "Bus", 500, 3)
    trip4 = Trip(paris, berlin, "Plane", 1200, 4)
    trip5 = Trip(paris, ny, "LuxuryJet", 5000, 5)
    trip6 = Trip(berlin, lisbon, "Train", 2100, 12)

    basic_trips = [trip1, trip2, trip3, trip4, trip5, trip6]

    # 3. DataFrame set up for calculations
    data_basic = []
    for t in basic_trips:
        data_basic.append({
            "Origin": t.origin.city,
            "Destination": t.destination.city,
            "Mode": t.transport_mode,
            "Cost": t.total_cost,
            "Days": t.duration_days,
            "Cost/Day": t.cost_per_day()
        })

    df_basic = pd.DataFrame(data_basic)

    # 4. Computing metrics requested by lab
    print("\n--- Summary Statistics (All Trips) ---")
    print(f"Total Cost of All Trips:      €{df_basic['Cost'].sum():,.2f}")
    print(f"Average Cost per Day:         €{df_basic['Cost/Day'].mean():,.2f}")
    print(f"Standard Deviation (Costs):   €{df_basic['Cost'].std():,.2f}")

    most_expensive = df_basic.loc[df_basic['Cost'].idxmax()]
    print(f"Most Expensive Trip:          {most_expensive['Origin']} -> {most_expensive['Destination']} (€{most_expensive['Cost']})")


    # ---------------------------------------------------------
    # PART 2: INTERACTIVE TRAVEL SYSTEM
    # ---------------------------------------------------------
    print("\n" + "#"*60)
    print(" PART 2: TRAVEL BOOKING INTERFACE")
    print("#"*60)

    # 1. User Inputs
    try:
        user_name = input("Please enter your name: ")
        budget_input = input("Please enter your total budget (€): ")
        user_budget = float(budget_input)
    except ValueError:
        print("Error: Please enter a valid number for the budget.")
        exit()

    print(f"\nWelcome, {user_name}. Searching for trips under €{user_budget:,.2f}...\n")

    # 2. Setup Advanced Inventory for EcoTrips() analysis
    tm_plane = TransportMode("Plane", 0.25, 850)
    tm_train = TransportMode("Train", 0.04, 300)
    tm_bus = TransportMode("Bus", 0.08, 100)
    tm_concorde = TransportMode("LuxuryJet", 0.40, 2000)

    # Inventory
    eco_trip1 = EcoTrip(lisbon, paris, tm_plane, 1600, 6, 1450)
    eco_trip2 = EcoTrip(paris, madrid, tm_train, 1800, 8, 1050)
    eco_trip3 = EcoTrip(lisbon, madrid, tm_bus, 500, 3, 500)
    eco_trip4 = EcoTrip(paris, berlin, tm_plane, 1200, 4, 880)
    eco_trip5 = EcoTrip(paris, ny, tm_concorde, 5000, 5, 5800)
    eco_trip6 = EcoTrip(berlin, lisbon, tm_train, 2100, 12, 2300)

    available_trips = [
        eco_trip1, eco_trip2, eco_trip3, eco_trip4, eco_trip5, eco_trip6]

    # 3. Filtering trips within budget
    affordable_options = []

    for trip in available_trips:
        if trip.total_cost <= user_budget:
            affordable_options.append({
                "Origin": trip.origin.city,
                "Destination": trip.destination.city,
                "Mode": trip.transport_mode.name,
                "Total Cost": trip.total_cost,
                "Duration (Days)": trip.duration_days,
                "Travel (h)": round(trip.est_travel_time_hours(), 1),
                "CO2 (kg)": round(trip.calculate_co2(), 2),
                "Status": "Affordable"
            })

    # 4. Display Results
    if not affordable_options:
        print(f"I'm sorry, {user_name}. No trips were found within your budget of €{user_budget:,.2f}.")
    else:
        df_options = pd.DataFrame(affordable_options)

        # Cheapest first
        df_options = df_options.sort_values(by="Total Cost")

        print("="*60)
        print(f" POSSIBLE BOOKINGS FOR: {user_name.upper()}")
        print(f" BUDGET LIMIT: €{user_budget:,.2f}")
        print("="*60)

        # Formatting
        format_mapping = {"Total Cost": "€{:,.2f}".format}

        print(df_options.to_string(index=False, formatters=format_mapping))

        print("-" * 60)
        print(f"Total options found: {len(df_options)}")

        # 5. Environmental Analysis (Best vs Worst)
        print("\n" + "="*60)
        print(" ENVIRONMENTAL IMPACT ANALYSIS")
        print("="*60)

        # Find Lowest CO2
        min_co2_idx = df_options["CO2 (kg)"].idxmin()
        best_trip = df_options.loc[min_co2_idx]

        # Find Highest CO2
        max_co2_idx = df_options["CO2 (kg)"].idxmax()
        worst_trip = df_options.loc[max_co2_idx]

        print("🌱 BEST CHOICE (Lowest CO2):")
        print(f"   {best_trip['Origin']} -> {best_trip['Destination']} ({best_trip['Mode']})")
        print(f"   Emissions: {best_trip['CO2 (kg)']} kg | Cost: €{best_trip['Total Cost']:,.2f}")

        print("\n⚠️  HIGHEST IMPACT (Highest CO2):")
        print(f"   {worst_trip['Origin']} -> {worst_trip['Destination']} ({worst_trip['Mode']})")
        print(f"   Emissions: {worst_trip['CO2 (kg)']} kg | Cost: €{worst_trip['Total Cost']:,.2f}")
        print("="*60)
