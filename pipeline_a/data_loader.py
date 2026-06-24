import os
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests


# Simple, beginner-friendly data loader for AgriFlow
# - Downloads commodity prices from Agmarknet (best-effort)
# - Downloads weather data from NASA POWER
# - Provides preprocessing (z-score normalization)


def _ensure_data_dir():
	"""Create the `data/` directory if it doesn't exist."""
	os.makedirs("data", exist_ok=True)


def _safe_request(url, params=None, timeout=10):
	"""Make a requests.get call with basic error handling.

	Returns the Response on success, or None on failure.
	"""
	try:
		resp = requests.get(url, params=params, timeout=timeout)
		resp.raise_for_status()
		return resp
	except requests.RequestException as e:
		# Print a simple error message for beginners and return None
		print(f"Request failed for {url} : {e}")
		return None

def get_mandi_coordinates(market_name, district, state):
	"""Geocode a mandi using OpenStreetMap Nominatim and return (lat, lon)."""
	if not market_name:
		return 20.5937, 78.9629

	clean_name = market_name
	if "(" in market_name:
		clean_name = market_name.split("(", 1)[0].strip()

	url = "https://nominatim.openstreetmap.org/search"
	headers = {"User-Agent": "AgriFlow/1.0"}
	params = {
		"q": f"{clean_name}, {district}, {state}, India",
		"format": "json",
		"limit": 1,
	}

	lat = None
	lon = None
	try:
		resp = requests.get(url, params=params, headers=headers, timeout=10)
		resp.raise_for_status()
		results = resp.json()
		if results:
			first = results[0]
			lat = float(first.get("lat"))
			lon = float(first.get("lon"))
	except requests.RequestException as e:
		print(f"Nominatim request failed: {e}")
		lat = None
		lon = None
	except (ValueError, TypeError):
		lat = None
		lon = None

	if lat is not None and lon is not None:
		print(f"Coordinates for {market_name}: {lat}, {lon}")
		return lat, lon

	# Respect rate limits before fallback query
	time.sleep(1)
	params["q"] = f"{district}, {state}, India"

	try:
		resp = requests.get(url, params=params, headers=headers, timeout=10)
		resp.raise_for_status()
		results = resp.json()
		if results:
			first = results[0]
			lat = float(first.get("lat"))
			lon = float(first.get("lon"))
	except requests.RequestException as e:
		print(f"Nominatim fallback request failed: {e}")
		lat = None
		lon = None
	except (ValueError, TypeError):
		lat = None
		lon = None

	if lat is not None and lon is not None:
		print(f"Coordinates for {market_name}: {lat}, {lon}")
		return lat, lon

	lat, lon = 20.5937, 78.9629
	print(f"Coordinates for {market_name}: {lat}, {lon}")
	return lat, lon

def get_agmarknet_data(commodity, state, days=90):
	"""Fetch commodity price data from Agmarknet for the given commodity and state.

	This function performs a best-effort GET request to an Agmarknet endpoint
	that returns JSON records. Agmarknet HTML pages and APIs vary by region
	and over time; if you have a specific documented endpoint or CSV URL,
	replace the `url` and parsing logic below accordingly.

	Args:
		commodity (str): commodity name, e.g. 'tomato'
		state (str): state name, e.g. 'Karnataka'
		days (int): number of past days to request (default 90)

	Returns:
		pandas.DataFrame: columns = [date, modal_price, min_price, max_price, arrival_volume, commodity, mandi]
	"""
	end_date = datetime.utcnow().date()
	start_date = end_date - timedelta(days=days)

	# Best-effort Agmarknet URL — update if you have a working endpoint.
	url = "https://agmarknet.gov.in/Price_Statistics/MarketWisePrices"
	params = {
		"commodity": commodity,
		"state": state,
		"start": start_date.strftime("%Y-%m-%d"),
		"end": end_date.strftime("%Y-%m-%d"),
		"format": "json",
	}

	resp = _safe_request(url, params=params)
	cols = ["date", "modal_price", "min_price", "max_price", "arrival_volume", "commodity", "mandi"]
	if resp is None:
		return pd.DataFrame(columns=cols)

	try:
		j = resp.json()
	except ValueError:
		print("Agmarknet response not JSON; can't parse.")
		return pd.DataFrame(columns=cols)

	# Locate a list of records in common keys
	records = None
	if isinstance(j, dict):
		for k in ("data", "records", "market", "prices"):
			if k in j and isinstance(j[k], list):
				records = j[k]
				break
	elif isinstance(j, list):
		records = j

	if not records:
		print("No records found in Agmarknet response.")
		return pd.DataFrame(columns=cols)

	rows = []
	for r in records:
		# extract fields safely with common fallbacks
		date = r.get("date") or r.get("Date") or r.get("price_date")
		modal = r.get("modal_price") or r.get("modal") or r.get("Modal_Price")
		min_p = r.get("min_price") or r.get("min") or r.get("Min_Price")
		max_p = r.get("max_price") or r.get("max") or r.get("Max_Price")
		arr = r.get("arrival_volume") or r.get("arrival") or r.get("Arrivals")
		mandi = r.get("market") or r.get("mandi") or r.get("Market")

		rows.append({
			"date": date,
			"modal_price": modal,
			"min_price": min_p,
			"max_price": max_p,
			"arrival_volume": arr,
			"commodity": commodity,
			"mandi": mandi,
		})

	df = pd.DataFrame(rows)
	# normalize types: convert date to datetime.date when possible
	if "date" in df.columns:
		df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

	_ensure_data_dir()
	filename = os.path.join("data", f"agmarknet_{commodity}_{state}.csv")
	df.to_csv(filename, index=False)

	return df


def get_weather_data(lat=None, lon=None, days=60, market_name=None, district=None, state=None):
	"""Fetch weather data from Open-Meteo API for a location.

	Uses the Open-Meteo forecast endpoint to retrieve daily temperature and
	precipitation values plus hourly relative humidity. The function returns a
	pandas DataFrame with columns: date, max_temp, min_temp, rainfall,
	humidity, drought_index, is_forecast.
	"""
	if market_name:
		if not district or not state:
			raise ValueError("district and state are required when market_name is provided")
		lat, lon = get_mandi_coordinates(market_name, district, state)
	elif lat is None or lon is None:
		raise ValueError("Either lat/lon or market_name must be provided")

	end_date = datetime.utcnow().date()
	start_date = end_date - timedelta(days=days)

	url = "https://api.open-meteo.com/v1/forecast"
	params = {
		"latitude": lat,
		"longitude": lon,
		"hourly": "relative_humidity_2m",
		"daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,et0_fao_evapotranspiration",
		"past_days": days,
		"forecast_days": 7,
		"timezone": "Asia/Kolkata",
	}

	resp = _safe_request(url, params=params)
	cols = ["date", "max_temp", "min_temp", "rainfall", "humidity", "drought_index", "is_forecast"]
	if resp is None:
		return pd.DataFrame(columns=cols)

	try:
		j = resp.json()
	except ValueError:
		print("Open-Meteo response not JSON; can't parse.")
		return pd.DataFrame(columns=cols)

	daily = j.get("daily", {})
	hourly = j.get("hourly", {})
	if not daily or "time" not in daily:
		print("Unexpected Open-Meteo response structure")
		return pd.DataFrame(columns=cols)

	daily_dates = daily.get("time", [])
	max_temp = daily.get("temperature_2m_max", [])
	min_temp = daily.get("temperature_2m_min", [])
	rainfall = daily.get("precipitation_sum", [])
	# hourly humidity values are used to compute a daily mean
	humidity_time = hourly.get("time", [])
	humidity_values = hourly.get("relative_humidity_2m", [])

	humidity = None
	if humidity_time and humidity_values:
		try:
			h_df = pd.DataFrame({
				"time": pd.to_datetime(humidity_time, errors="coerce"),
				"humidity": pd.to_numeric(humidity_values, errors="coerce"),
			})
			h_df["date"] = h_df["time"].dt.date
			humidity = h_df.groupby("date")["humidity"].mean().to_dict()
		except Exception:
			humidity = {}
	else:
		humidity = {}

	rows = []
	for idx, date_str in enumerate(daily_dates):
		try:
			row_date = datetime.strptime(date_str, "%Y-%m-%d").date()
		except Exception:
			row_date = None

		rows.append({
			"date": row_date,
			"max_temp": max_temp[idx] if idx < len(max_temp) else None,
			"min_temp": min_temp[idx] if idx < len(min_temp) else None,
			"rainfall": rainfall[idx] if idx < len(rainfall) else None,
			"humidity": humidity.get(row_date) if row_date is not None else None,
			"drought_index": None,
		})

	df = pd.DataFrame(rows)
	if "rainfall" in df.columns:
		df["rainfall"] = pd.to_numeric(df["rainfall"], errors="coerce")
		df["drought_index"] = df["rainfall"].apply(lambda x: 1 if pd.notna(x) and x < 2.0 else 0)

	# Mark historical vs forecast rows: first 'days' rows are historical, rest are forecast
	df["is_forecast"] = 0
	if len(df) > days:
		df.loc[days:, "is_forecast"] = 1

	# Count rows
	n_historical = (df["is_forecast"] == 0).sum()
	n_forecast = (df["is_forecast"] == 1).sum()
	print(f"Weather data: {n_historical} historical + {n_forecast} forecast rows")

	_ensure_data_dir()
	filename = os.path.join("data", f"weather_{lat}_{lon}.csv")
	df.to_csv(filename, index=False)

	return df


def preprocess_and_normalize(df):
	"""Z-score normalize all numeric columns in the DataFrame.

	The function leaves non-numeric columns unchanged and returns a new
	DataFrame. Standard score is computed as (x - mean) / std.
	If a column has zero standard deviation it will be set to 0.0.
	"""
	df = df.copy()
	numeric_cols = df.select_dtypes(include=[np.number]).columns
	# Skip normalization for is_forecast column (should remain 0 or 1)
	cols_to_normalize = [c for c in numeric_cols if c != "is_forecast"]
	for c in cols_to_normalize:
		col = pd.to_numeric(df[c], errors="coerce")
		mean = col.mean()
		std = col.std(ddof=0)
		if std == 0 or np.isnan(std):
			df[c] = 0.0
		else:
			df[c] = (col - mean) / std
	return df


if __name__ == "__main__":
	# Example usage: fetch a small sample and print shapes. Update commodity/state as needed.
	print("Example: fetching tomato prices (best-effort) and weather for a point...")
	df_prices = get_agmarknet_data("tomato", "Karnataka", days=30)
	print("Prices:", df_prices.shape)
	df_weather = get_weather_data(12.97, 77.59, days=30)  # Bangalore lat/lon
	print("Weather:", df_weather.shape)
	# Demonstrate normalization
	if not df_weather.empty:
		df_weather_norm = preprocess_and_normalize(df_weather.select_dtypes(include=[np.number]))
		print("Normalized weather numeric columns:\n", df_weather_norm.head())

	df_weather_lasalgaon = get_weather_data(market_name="Lasalgaon(Niphad)", district="Nashik", state="Maharashtra")
	print("Weather for Lasalgaon:", df_weather_lasalgaon.shape)
	df_weather_bangalore = get_weather_data(market_name="Bangalore Central", district="Bangalore", state="Karnataka")
	print("Weather for Bangalore Central:", df_weather_bangalore.shape)

