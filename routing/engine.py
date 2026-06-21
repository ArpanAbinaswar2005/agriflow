from typing import List, Dict, Tuple, Optional
import numpy as np


def compute_routing_score(
    price_forecast: float,
    deficit_score: float,
    transport_cost: float,
    provisional_grade: str,
    trust_score: float,
) -> float:
    """Compute the mandi recommendation score.

    Implements the proposal formula:
        S(i,m) = w1*price_forecast + w2*deficit_score - w3*transport_cost
                  + w4*grade_premium + w5*trust_score

    Where:
        w1=0.35 (price importance)
        w2=0.25 (supply deficit importance)
        w3=0.20 (transport cost penalty)
        w4=0.15 (quality grade premium)
        w5=0.05 (farmer trust bonus)

    Args:
        price_forecast (float): forecasted modal price for the crop
        deficit_score (float): weather-adjusted supply deficit (0-1)
        transport_cost (float): estimated transport cost to mandi
        provisional_grade (str): quality grade 'A', 'B', or 'C'
        trust_score (float): farmer's historical accuracy (0-1)

    Returns:
        float: Combined routing score for the mandi
    """
    # Weights
    w1, w2, w3, w4, w5 = 0.35, 0.25, 0.20, 0.15, 0.05

    # Grade premium mapping
    grade_premium = {"A": 0.15, "B": 0.0, "C": -0.20}
    premium = grade_premium.get(provisional_grade.upper(), 0.0)

    # Compute score
    score = (
        w1 * price_forecast
        + w2 * deficit_score
        - w3 * transport_cost
        + w4 * premium
        + w5 * trust_score
    )

    return score


def compute_weather_deficit(
    forecasted_arrivals: float,
    historical_mean: float,
    weather_stress_index: float,
) -> float:
    """Compute weather-adjusted supply deficit score.

    Implements the proposal deficit formula:
        Dc,m,t = (1 - arrivals/mean) * (1 + stress)

    High deficit (close to 1) means low supply -> good opportunity for farmer.
    Low deficit (close to 0) means high supply -> less opportunity.

    Args:
        forecasted_arrivals (float): predicted crop arrivals (tons)
        historical_mean (float): historical average arrivals (tons)
        weather_stress_index (float): weather stress factor (0-1)
                                      0=no stress, 1=extreme stress

    Returns:
        float: Deficit score in range [0, ~2] (higher = better opportunity)
    """
    if historical_mean <= 0:
        return 0.0

    # Base deficit: (1 - arrivals/mean)
    base_deficit = 1.0 - (forecasted_arrivals / historical_mean)
    base_deficit = np.clip(base_deficit, 0, 1)  # Clamp to [0, 1]

    # Weather adjustment: high stress increases opportunity
    deficit_score = base_deficit * (1.0 + weather_stress_index)

    return float(deficit_score)


class MandiRecommender:
    """Mandi recommendation engine based on multi-factor scoring.

    Scores mandis using price forecasts, supply deficits, transport costs,
    quality grades, and farmer trust. Returns top-k ranked recommendations.
    """

    def __init__(self, mandis: List[Dict]):
        """Initialize the recommender with mandi data.

        Args:
            mandis: List of dicts with keys:
                - mandi_name (str): unique identifier
                - location (str): geographic location
                - latitude (float): latitude for distance calc
                - longitude (float): longitude for distance calc
                - historical_avg_arrivals (float): historical inflow (tons)
        """
        self.mandis = mandis

    def _estimate_transport_cost(self, farmer_lat: float, farmer_lon: float, mandi: Dict) -> float:
        """Estimate transport cost based on distance.

        Simple model: cost = distance * rate (e.g., 10 rupees per km)
        Distance computed using Euclidean approximation for small distances.

        Args:
            farmer_lat, farmer_lon: farmer's coordinates
            mandi: mandi dict with latitude/longitude

        Returns:
            float: Estimated transport cost (normalized to [0,1] for scoring)
        """
        # Euclidean distance (simplified for demo)
        dlat = farmer_lat - mandi["latitude"]
        dlon = farmer_lon - mandi["longitude"]
        distance_km = np.sqrt(dlat**2 + dlon**2) * 111  # ~111 km per degree
        
        # Cost: 10 rupees per km
        cost_rupees = distance_km * 10.0
        
        # Normalize to [0, 1] assuming max cost ~5000 rupees (500 km)
        normalized_cost = min(cost_rupees / 5000.0, 1.0)
        
        return normalized_cost

    def recommend(
        self,
        farmer_lat: float,
        farmer_lon: float,
        price_forecasts: Dict[str, float],  # {mandi_name: price}
        weather_stress_index: float,  # 0-1
        trust_scores: Dict[str, float],  # {mandi_name: trust_score}
        provisional_grade: str,
        top_k: int = 3,
    ) -> List[Dict]:
        """Recommend top-k mandis for the farmer's crop.

        Args:
            farmer_lat, farmer_lon: farmer's location
            price_forecasts: dict mapping mandi names to forecasted prices
            weather_stress_index: weather stress factor (0-1)
            trust_scores: dict mapping mandi names to trust scores (0-1)
            provisional_grade: quality grade 'A', 'B', or 'C'
            top_k: number of recommendations to return

        Returns:
            List of dicts (sorted by score, descending) with keys:
                - rank (int): 1, 2, 3...
                - mandi_name (str)
                - location (str)
                - score (float): routing score
                - estimated_price (float)
                - transport_cost (float): normalized
                - deficit_score (float)
                - trust_score (float)
        """
        recommendations = []

        for mandi in self.mandis:
            mandi_name = mandi["mandi_name"]

            # Get values; use defaults if not available
            price_forecast = price_forecasts.get(mandi_name, 50.0)
            trust_score = trust_scores.get(mandi_name, 0.5)

            # Transport cost
            transport_cost = self._estimate_transport_cost(farmer_lat, farmer_lon, mandi)

            # Supply deficit (simplified: assume 70% of historical mean as forecast)
            forecasted_arrivals = mandi["historical_avg_arrivals"] * 0.7
            deficit_score = compute_weather_deficit(
                forecasted_arrivals,
                mandi["historical_avg_arrivals"],
                weather_stress_index,
            )

            # Compute routing score
            routing_score = compute_routing_score(
                price_forecast=price_forecast,
                deficit_score=deficit_score,
                transport_cost=transport_cost,
                provisional_grade=provisional_grade,
                trust_score=trust_score,
            )

            recommendations.append({
                "mandi_name": mandi_name,
                "location": mandi["location"],
                "score": routing_score,
                "estimated_price": price_forecast,
                "transport_cost": transport_cost,
                "deficit_score": deficit_score,
                "trust_score": trust_score,
            })

        # Sort by score (descending)
        recommendations.sort(key=lambda x: x["score"], reverse=True)

        # Add rank and return top_k
        for i, rec in enumerate(recommendations[:top_k], start=1):
            rec["rank"] = i

        return recommendations[:top_k]


if __name__ == "__main__":
    print("=== AgriFlow Routing Engine Test ===\n")

    # Define 5 sample mandis
    mandis = [
        {
            "mandi_name": "Bangalore Central",
            "location": "Bangalore",
            "latitude": 12.97,
            "longitude": 77.59,
            "historical_avg_arrivals": 500.0,
        },
        {
            "mandi_name": "Yeshwanthpur Market",
            "location": "Bangalore North",
            "latitude": 13.35,
            "longitude": 77.58,
            "historical_avg_arrivals": 300.0,
        },
        {
            "mandi_name": "Kolar Mandi",
            "location": "Kolar",
            "latitude": 13.15,
            "longitude": 78.13,
            "historical_avg_arrivals": 200.0,
        },
        {
            "mandi_name": "Tumkur Market",
            "location": "Tumkur",
            "latitude": 13.22,
            "longitude": 77.10,
            "historical_avg_arrivals": 350.0,
        },
        {
            "mandi_name": "Chikballapur Mandi",
            "location": "Chikballapur",
            "latitude": 13.45,
            "longitude": 77.85,
            "historical_avg_arrivals": 250.0,
        },
    ]

    # Farmer location (Delhi for demo)
    farmer_lat, farmer_lon = 28.70, 77.10

    # Sample inputs: price forecasts for each mandi (from Pipeline A)
    price_forecasts = {
        "Bangalore Central": 65.0,
        "Yeshwanthpur Market": 62.0,
        "Kolar Mandi": 58.0,
        "Tumkur Market": 70.0,
        "Chikballapur Mandi": 61.0,
    }

    # Sample trust scores (from farmer's historical accuracy)
    trust_scores = {
        "Bangalore Central": 0.8,
        "Yeshwanthpur Market": 0.6,
        "Kolar Mandi": 0.5,
        "Tumkur Market": 0.7,
        "Chikballapur Mandi": 0.65,
    }

    # Weather stress (0=no stress, 1=extreme)
    weather_stress = 0.3  # Moderate stress

    # Provisional grade from Pipeline B
    provisional_grade = "A"

    # Create recommender and generate recommendations
    recommender = MandiRecommender(mandis)
    recommendations = recommender.recommend(
        farmer_lat=farmer_lat,
        farmer_lon=farmer_lon,
        price_forecasts=price_forecasts,
        weather_stress_index=weather_stress,
        trust_scores=trust_scores,
        provisional_grade=provisional_grade,
        top_k=3,
    )

    print(f"Farmer location: ({farmer_lat}, {farmer_lon})")
    print(f"Crop grade: {provisional_grade}")
    print(f"Weather stress: {weather_stress}\n")

    print("Top 3 Mandi Recommendations:")
    print("-" * 80)
    for rec in recommendations:
        print(f"Rank {rec['rank']}: {rec['mandi_name']} ({rec['location']})")
        print(f"  Score: {rec['score']:.4f}")
        print(f"  Est. Price: ₹{rec['estimated_price']:.2f}/unit")
        print(f"  Transport Cost: {rec['transport_cost']:.4f} (normalized)")
        print(f"  Deficit Score: {rec['deficit_score']:.4f}")
        print(f"  Trust Score: {rec['trust_score']:.2f}")
        print()
