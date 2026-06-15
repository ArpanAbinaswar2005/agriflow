import os
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


def get_weather_data(lat, lon, days=60):
	"""Fetch weather data from NASA POWER API for a location.

	Uses the NASA POWER daily point API to retrieve precipitation (rainfall),
	daily max temperature, daily min temperature, and relative humidity (if available).

	The function returns a pandas DataFrame with columns:
		date, rainfall, max_temp, min_temp, humidity, drought_index

	The `drought_index` is a simple, relative indicator computed as the negative
	z-score of rainfall across the returned period (so higher values indicate drier conditions).
	"""
	end_date = datetime.utcnow().date()
	start_date = end_date - timedelta(days=days)

	url = "https://power.larc.nasa.gov/api/temporal/daily/point"
	params = {
		"parameters": "PRECTOT,T2M_MAX,T2M_MIN,RH2M",
		"community": "AG",
		"latitude": lat,
		"longitude": lon,
		"start": start_date.strftime("%Y%m%d"),
		"end": end_date.strftime("%Y%m%d"),
		"format": "JSON",
	}

	resp = _safe_request(url, params=params)
	cols = ["date", "rainfall", "max_temp", "min_temp", "humidity", "drought_index"]
	if resp is None:
		return pd.DataFrame(columns=cols)

	try:
		j = resp.json()
	except ValueError:
		print("NASA POWER response not JSON; can't parse.")
		return pd.DataFrame(columns=cols)

	try:
		params_data = j["properties"]["parameter"]
	except Exception:
		print("Unexpected NASA POWER response structure")
		return pd.DataFrame(columns=cols)

	rainfall = params_data.get("PRECTOT", {})
	tmax = params_data.get("T2M_MAX", {})
	tmin = params_data.get("T2M_MIN", {})
	rh = params_data.get("RH2M", {})

	rows = []
	for d_str, r_val in rainfall.items():
		try:
			d = datetime.strptime(d_str, "%Y%m%d").date()
		except Exception:
			d = None

		rows.append({
			"date": d,
			"rainfall": r_val,
			"max_temp": tmax.get(d_str),
			"min_temp": tmin.get(d_str),
			"humidity": rh.get(d_str),
			"drought_index": None,
		})

	df = pd.DataFrame(rows)

	# Convert numeric columns and compute drought index
	if "rainfall" in df.columns and not df["rainfall"].isnull().all():
		r = pd.to_numeric(df["rainfall"], errors="coerce")
		std = r.std(ddof=0)
		std = std if std != 0 else 1.0
		df["drought_index"] = -((r - r.mean()) / std)

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
	for c in numeric_cols:
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

