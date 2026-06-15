import os
import sys
from datetime import datetime
from pathlib import Path

# Ensure the project root is on sys.path when running train.py directly.
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from pipeline_a.data_loader import get_agmarknet_data, get_weather_data, preprocess_and_normalize
from pipeline_a.model import AgriFlowPipelineA


def _ensure_models_dir():
    """Create the models directory if it does not exist."""
    os.makedirs("models", exist_ok=True)


def generate_synthetic_data():
    """Create simple sample price and weather data for testing."""
    price_days = 120
    weather_days = 90
    price_dates = pd.date_range(end=datetime.now().date(), periods=price_days)
    weather_dates = pd.date_range(end=datetime.now().date(), periods=weather_days)

    # Simple synthetic price series that trends upward with a little noise
    price_base = np.linspace(50, 80, price_days)
    price_noise = np.random.randn(price_days) * 2.0
    modal_price = price_base + price_noise
    min_price = modal_price - np.abs(np.random.randn(price_days) * 2.0)
    max_price = modal_price + np.abs(np.random.randn(price_days) * 2.0)
    arrival_volume = np.linspace(100, 150, price_days) + np.random.randn(price_days) * 5.0

    df_price = pd.DataFrame(
        {
            "date": price_dates,
            "modal_price": modal_price,
            "min_price": min_price,
            "max_price": max_price,
            "arrival_volume": arrival_volume,
            "commodity": ["tomato"] * price_days,
            "mandi": ["SampleMandi"] * price_days,
        }
    )

    # Simple synthetic weather data with season-like changes
    weather_base = np.linspace(10, 30, weather_days)
    rainfall = np.abs(np.sin(np.linspace(0, 3.0, weather_days)) * 5.0 + np.random.randn(weather_days))
    max_temp = weather_base + np.random.randn(weather_days) * 1.5
    min_temp = max_temp - np.abs(np.random.randn(weather_days) * 3.0)
    humidity = np.clip(50 + np.cos(np.linspace(0, 3.0, weather_days)) * 20 + np.random.randn(weather_days) * 3.0, 10, 100)
    drought_index = -((rainfall - rainfall.mean()) / (rainfall.std(ddof=0) if rainfall.std(ddof=0) != 0 else 1.0))

    df_weather = pd.DataFrame(
        {
            "date": weather_dates,
            "rainfall": rainfall,
            "max_temp": max_temp,
            "min_temp": min_temp,
            "humidity": humidity,
            "drought_index": drought_index,
        }
    )

    return df_price, df_weather


def load_sample_data():
    """Load sample price and weather data from disk or generate synthetic data."""
    _ensure_models_dir()
    price_path = os.path.join("data", "agmarknet_tomato_Karnataka.csv")
    weather_path = os.path.join("data", "weather_12.97_77.59.csv")

    if os.path.exists(price_path) and os.path.exists(weather_path):
        print("Loading sample data from disk.")
        df_price = pd.read_csv(price_path, parse_dates=["date"])
        df_weather = pd.read_csv(weather_path, parse_dates=["date"])
        return df_price, df_weather

    print("Sample data not found on disk, generating synthetic data.")
    df_price, df_weather = generate_synthetic_data()
    return df_price, df_weather


def prepare_sequences(df_price: pd.DataFrame, df_weather: pd.DataFrame):
    """Convert price and weather dataframes into training tensors.

    The function returns:
        price_tensor: (samples, 90, 5)
        weather_tensor: (samples, 60, 5)
        target_tensor: (samples, 3)
    """
    # Copy and sort by date if available
    df_price = df_price.copy()
    df_weather = df_weather.copy()
    if "date" in df_price.columns:
        df_price = df_price.sort_values("date")
    if "date" in df_weather.columns:
        df_weather = df_weather.sort_values("date")

    # Add a numeric time feature for price if date exists
    if "date" in df_price.columns:
        df_price["day_of_year"] = df_price["date"].dt.dayofyear
    else:
        df_price["day_of_year"] = 0

    # Normalize numeric columns before creating sequences
    df_price = preprocess_and_normalize(df_price)
    df_weather = preprocess_and_normalize(df_weather)

    price_columns = ["modal_price", "min_price", "max_price", "arrival_volume", "day_of_year"]
    weather_columns = ["rainfall", "max_temp", "min_temp", "humidity", "drought_index"]

    for col in price_columns:
        if col not in df_price.columns:
            df_price[col] = 0.0
    for col in weather_columns:
        if col not in df_weather.columns:
            df_weather[col] = 0.0

    price_values = df_price[price_columns].fillna(0.0).values
    weather_values = df_weather[weather_columns].fillna(0.0).values
    modal_values = df_price["modal_price"].fillna(0.0).values

    seq_price = 90
    seq_weather = 60
    sample_count = min(len(price_values) - seq_price + 1, len(weather_values) - seq_weather + 1)
    sample_count = max(sample_count, 0)

    if sample_count == 0:
        raise ValueError("Not enough data to create sample sequences. Need at least 90 price rows and 60 weather rows.")

    price_samples = []
    weather_samples = []
    targets = []

    for i in range(sample_count):
        price_slice = price_values[i : i + seq_price]
        weather_slice = weather_values[i : i + seq_weather]
        price_samples.append(price_slice)
        weather_samples.append(weather_slice)

        # Create simple proxy targets for 7d, 14d, and 30d price forecasts.
        # If future data is available, use the next modal price averages.
        if i + seq_price + 30 <= len(modal_values):
            t7 = float(np.mean(modal_values[i + seq_price : i + seq_price + 7]))
            t14 = float(np.mean(modal_values[i + seq_price : i + seq_price + 14]))
            t30 = float(np.mean(modal_values[i + seq_price : i + seq_price + 30]))
        else:
            next_base = float(modal_values[i + seq_price - 1])
            t7 = next_base + 0.01
            t14 = next_base + 0.02
            t30 = next_base + 0.03

        targets.append([t7, t14, t30])

    price_tensor = torch.tensor(np.stack(price_samples), dtype=torch.float32)
    weather_tensor = torch.tensor(np.stack(weather_samples), dtype=torch.float32)
    target_tensor = torch.tensor(np.stack(targets), dtype=torch.float32)

    return price_tensor, weather_tensor, target_tensor


def train_model(model, train_loader, val_loader, epochs=50):
    """Train the model and return train/validation loss lists."""
    _ensure_models_dir()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    best_val_loss = float("inf")
    best_path = os.path.join("models", "pipeline_a_best.pt")
    train_losses = []
    val_losses = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        for price_batch, weather_batch, target_batch in train_loader:
            price_batch = price_batch.to(device)
            weather_batch = weather_batch.to(device)
            target_batch = target_batch.to(device)

            optimizer.zero_grad()
            predictions = model(price_batch, weather_batch)
            loss = loss_fn(predictions, target_batch)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * price_batch.size(0)

        avg_train_loss = total_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for price_batch, weather_batch, target_batch in val_loader:
                price_batch = price_batch.to(device)
                weather_batch = weather_batch.to(device)
                target_batch = target_batch.to(device)
                predictions = model(price_batch, weather_batch)
                loss = loss_fn(predictions, target_batch)
                total_val_loss += loss.item() * price_batch.size(0)

        avg_val_loss = total_val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_path)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

    print(f"Best validation loss saved to {best_path}")
    return train_losses, val_losses


def plot_losses(train_losses, val_losses):
    """Plot loss curves and save them to the models folder."""
    _ensure_models_dir()
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join("models", "loss_curve.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved loss curve to {plot_path}")


def evaluate_model(model, data_loader):
    """Compute mean squared error on a given data loader."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    loss_fn = torch.nn.MSELoss()
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for price_batch, weather_batch, target_batch in data_loader:
            price_batch = price_batch.to(device)
            weather_batch = weather_batch.to(device)
            target_batch = target_batch.to(device)
            predictions = model(price_batch, weather_batch)
            loss = loss_fn(predictions, target_batch)
            total_loss += loss.item() * price_batch.size(0)
    return total_loss / len(data_loader.dataset)


def split_data(price_tensor, weather_tensor, target_tensor):
    """Split tensors into train/val/test data loaders."""
    total = price_tensor.size(0)
    indices = np.arange(total)
    np.random.shuffle(indices)

    train_end = int(total * 0.70)
    val_end = int(total * 0.85)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    train_dataset = TensorDataset(price_tensor[train_idx], weather_tensor[train_idx], target_tensor[train_idx])
    val_dataset = TensorDataset(price_tensor[val_idx], weather_tensor[val_idx], target_tensor[val_idx])
    test_dataset = TensorDataset(price_tensor[test_idx], weather_tensor[test_idx], target_tensor[test_idx])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader, test_loader


def main():
    """Main script entry point for training Pipeline A."""
    df_price, df_weather = load_sample_data()
    price_tensor, weather_tensor, target_tensor = prepare_sequences(df_price, df_weather)

    train_loader, val_loader, test_loader = split_data(price_tensor, weather_tensor, target_tensor)

    model = AgriFlowPipelineA(price_input_size=5, weather_input_size=5)
    train_losses, val_losses = train_model(model, train_loader, val_loader, epochs=50)
    plot_losses(train_losses, val_losses)

    # Load best saved model and evaluate on test data
    best_model = AgriFlowPipelineA(price_input_size=5, weather_input_size=5)
    best_path = os.path.join("models", "pipeline_a_best.pt")
    best_model.load_state_dict(torch.load(best_path, map_location=torch.device("cpu")))
    test_mse = evaluate_model(best_model, test_loader)
    print(f"Final test MSE: {test_mse:.6f}")


if __name__ == "__main__":
    main()
