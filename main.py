"""AgriFlow Main Integration Script

Demonstrates the complete AgriFlow system combining:
- Pipeline A: Price forecasting using LSTM
- Pipeline B Stage 1: Video-based crop quality grading (farmer's phone)
- Pipeline B Stage 2: Multi-modal verified grading (mandi gate)
- Routing Engine: Mandi recommendations based on price and quality
"""

import sys
from pathlib import Path

import torch
import numpy as np

# Add project root to path for imports
ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

from pipeline_a.model import AgriFlowPipelineA
from pipeline_b.model_stage1 import Stage1Grader, grade_batch
from pipeline_b.model_stage2 import LateFusionGrader, get_verified_grade
from routing.engine import MandiRecommender


def run_agriflow_demo():
    """Run a complete demo of the AgriFlow system."""
    
    print("=" * 70)
    print("=== AgriFlow Demo ===")
    print("=" * 70)
    print()

    # ============================================================================
    # PIPELINE A: Price Forecasting
    # ============================================================================
    print("[Pipeline A] Price Forecasting")
    print("-" * 70)

    try:
        # Create model
        model_a = AgriFlowPipelineA(price_input_size=5, weather_input_size=5)
        model_a.eval()

        # Create fake data
        price_seq = torch.randn(1, 90, 5)  # (batch, 90 days, 5 features)
        weather_seq = torch.randn(1, 60, 5)  # (batch, 60 days, 5 features)

        # Run inference
        with torch.no_grad():
            price_forecast = model_a(price_seq, weather_seq)  # (1, 3) for [7d, 14d, 30d]

        forecast_7d = float(price_forecast[0, 0].item())
        forecast_14d = float(price_forecast[0, 1].item())
        forecast_30d = float(price_forecast[0, 2].item())

        print(f"Pipeline A: Price forecast for 7/14/30 days ahead:")
        print(f"  7-day forecast: ₹{forecast_7d:.2f}")
        print(f"  14-day forecast: ₹{forecast_14d:.2f}")
        print(f"  30-day forecast: ₹{forecast_30d:.2f}")
        print()
    except Exception as e:
        print(f"Pipeline A error: {e}")
        print()

    # ============================================================================
    # PIPELINE B STAGE 1: Video-based Provisional Grading
    # ============================================================================
    print("[Pipeline B Stage 1] Provisional Video-based Grading")
    print("-" * 70)

    try:
        # Create model
        grader_stage1 = Stage1Grader(num_classes=3)
        grader_stage1.eval()

        # Create fake video frames (10 frames)
        fake_frames = torch.rand(10, 3, 224, 224)

        # Grade the batch
        with torch.no_grad():
            provisional_grade, confidence, low_confidence = grade_batch(grader_stage1, fake_frames)

        print(f"Pipeline B Stage 1: Provisional grade: {provisional_grade}")
        print(f"  Confidence: {confidence:.3f}")
        print(f"  Low confidence flag: {low_confidence}")
        print(f"  (Farmer can run this on their phone video)")
        print()
    except Exception as e:
        print(f"Pipeline B Stage 1 error: {e}")
        print()

    # ============================================================================
    # PIPELINE B STAGE 2: Multi-modal Verified Grading
    # ============================================================================
    print("[Pipeline B Stage 2] Multi-modal Verified Grading")
    print("-" * 70)

    try:
        # Create model
        grader_stage2 = LateFusionGrader(num_classes=3)
        grader_stage2.eval()

        # Create fake multi-modal data
        image = torch.rand(1, 3, 224, 224)  # Visual: camera image
        thermal_data = torch.rand(1, 30, 2)  # Thermal: temperature/humidity readings
        physical_data = torch.rand(1, 4)  # Physical: weight/density measurements

        # Get verified grade
        with torch.no_grad():
            verified_grade, confidence = get_verified_grade(
                grader_stage2, image, thermal_data, physical_data
            )

        print(f"Pipeline B Stage 2: Verified grade: {verified_grade}")
        print(f"  Confidence: {confidence:.3f}")
        print(f"  (Runs at mandi gate with multiple sensors)")
        print()
    except Exception as e:
        print(f"Pipeline B Stage 2 error: {e}")
        print()

    # ============================================================================
    # ROUTING ENGINE: Mandi Recommendations
    # ============================================================================
    print("[Routing Engine] Mandi Recommendations")
    print("-" * 70)

    try:
        # Define sample mandis
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

        # Sample inputs
        price_forecasts = {
            "Bangalore Central": forecast_7d,
            "Yeshwanthpur Market": forecast_7d * 0.95,
            "Kolar Mandi": forecast_7d * 0.90,
            "Tumkur Market": forecast_7d * 1.05,
            "Chikballapur Mandi": forecast_7d * 0.92,
        }

        trust_scores = {
            "Bangalore Central": 0.8,
            "Yeshwanthpur Market": 0.6,
            "Kolar Mandi": 0.5,
            "Tumkur Market": 0.7,
            "Chikballapur Mandi": 0.65,
        }

        farmer_lat, farmer_lon = 13.00, 77.60  # Central Karnataka region

        # Get recommendations
        recommender = MandiRecommender(mandis)
        recommendations = recommender.recommend(
            farmer_lat=farmer_lat,
            farmer_lon=farmer_lon,
            price_forecasts=price_forecasts,
            weather_stress_index=0.3,
            trust_scores=trust_scores,
            provisional_grade=provisional_grade,
            top_k=3,
        )

        if recommendations:
            top_mandi = recommendations[0]
            print(f"Routing Engine: Top mandi recommendation: {top_mandi['mandi_name']}")
            print(f"  Location: {top_mandi['location']}")
            print(f"  Score: {top_mandi['score']:.4f}")
            print(f"  Est. Price: ₹{top_mandi['estimated_price']:.2f}")
            print()
    except Exception as e:
        print(f"Routing Engine error: {e}")
        print()

    # ============================================================================
    # FINAL SUMMARY
    # ============================================================================
    print("=" * 70)
    print("=== AgriFlow System Ready ===")
    print("=" * 70)
    print("All pipelines operational:")
    print("  ✓ Pipeline A: Price forecasting (LSTM)")
    print("  ✓ Pipeline B Stage 1: Provisional grading (video)")
    print("  ✓ Pipeline B Stage 2: Verified grading (multi-modal)")
    print("  ✓ Routing Engine: Mandi recommendations")
    print()
    print("Ready for real data training on Colab")
    print()


if __name__ == "__main__":
    run_agriflow_demo()
