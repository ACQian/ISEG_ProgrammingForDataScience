# Lab_01: Travel Analytics Lab: Modeling Trips and Places

import pandas as pd

# Instruction: The first class, Place, represents a city or location,
# and should store attributes such as the name of the city and the country.


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


# Instruction: The second class, Trip, represents a journey connecting
# two Place objects. Each Trip object should include the origin and
# destination (both Place objects), as well as attributes such as transport
# mode, total cost, and duration in days.
# You should also implement a method to compute the cost per day for the trip.


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


# Instruction: Create multiple Place objects representing different cities,
# and several Trip objects connecting these places.
# All trips should be stored in a list.
# Using this structure, compute metrics such as the total cost of all trips,
# the average cost per day, the most expensive trip,
# and the standard deviation of trip costs.

if __name__ == "__main__":
    # Creating Place objects
    lisbon = Place("Lisbon", "Portugal")
    paris = Place("Paris", "France")
    madrid = Place("Madrid", "Spain")

    # Creating Trip objects
    trip1 = Trip(lisbon, paris, "Plane", 1600, 6)
    trip2 = Trip(paris, madrid, "Train", 1800, 8)
    trip3 = Trip(lisbon, madrid, "Bus", 500, 3)

    # Storing trips in a list
    trips = [trip1, trip2, trip3]

    # Flattening objects into DF
    data = []
    for trip in trips:
        data.append({
            "Origin_City": trip.origin.city,
            "Origin_Country": trip.origin.country,
            "Destination_City": trip.destination.city,
            "Destination_Country": trip.destination.country,
            "Transport_Mode": trip.transport_mode,
            "Total_Cost": trip.total_cost,
            "Duration_Days": trip.duration_days,
            "Cost_per_Day": trip.cost_per_day()
        })

    df = pd.DataFrame(data)

    # Total cost of all trips
    total_cost_all_trips = df["Total_Cost"].sum()
    print("Total cost of all trips:", total_cost_all_trips)

    # Average cost per day
    average_cost_per_day = df["Cost_per_Day"].mean()
    print("Average cost per day:", average_cost_per_day)

    # Most expensive trip
    most_expensive_trip = df.loc[df["Total_Cost"].idxmax()]
    print(
        "Most expensive trip:", most_expensive_trip["Origin_City"],
        "to", most_expensive_trip["Destination_City"])

    # Standard deviation of trip costs
    std_dev_trip_costs = df["Total_Cost"].std()
    print("Standard deviation of trip costs:", std_dev_trip_costs)
