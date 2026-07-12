"""Service 1: API-backed weather service using Open-Meteo.

The service makes live HTTP API calls and transforms the structured response into
short natural-language text instead of returning raw JSON.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import requests

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
TIMEOUT_SECONDS = 10

WEATHER_CODE_DESCRIPTIONS = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    56: "light freezing drizzle",
    57: "dense freezing drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    66: "light freezing rain",
    67: "heavy freezing rain",
    71: "slight snow fall",
    73: "moderate snow fall",
    75: "heavy snow fall",
    77: "snow grains",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    85: "slight snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with slight hail",
    99: "thunderstorm with heavy hail",
}


@dataclass
class Location:
    name: str
    country: str
    latitude: float
    longitude: float
    timezone: str


class WeatherService:
    def extract_city(self, message: str) -> str:
        """Best-effort city extraction from free text or slash commands."""
        text = message.strip()
        text = re.sub(r"^/(weather|api)\s*", "", text, flags=re.IGNORECASE).strip()
        patterns = [
            r"(?:weather|forecast|temperature|气温|天气|预报)\s+(?:in|for|at|of)?\s*([A-Za-z\u4e00-\u9fff .,'-]{2,})",
            r"(?:in|for|at)\s+([A-Za-z\u4e00-\u9fff .,'-]{2,})\s*(?:today|now|right now)?",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                city = match.group(1).strip(" ?.!,")
                city = re.sub(r"\b(today|now|right now|please|pls)\b", "", city, flags=re.IGNORECASE).strip()
                if city:
                    return city
        # If the command removed the keyword and left a short phrase, treat it as the city.
        if 2 <= len(text) <= 80 and not re.search(r"\?", text):
            return text.strip(" ?.!,")
        return ""

    def answer(self, message: str) -> str:
        city = self.extract_city(message)
        if not city:
            return "Tell me a city, and I’ll turn the weather API response into a short plain-English update. Example: `/weather Toronto`."
        try:
            location = self._geocode(city)
            if location is None:
                return f"I couldn’t find a matching city for **{city}**. Try a more specific name, such as `Paris, France` or `Toronto`."
            current = self._fetch_current_weather(location)
            return self._format_weather(location, current)
        except requests.RequestException as exc:
            return f"The weather API did not respond successfully: {exc}. Try again after checking your network connection."
        except (KeyError, TypeError, ValueError) as exc:
            return f"The weather API returned an unexpected shape, so I could not summarize it safely. Details: {exc}."

    def _geocode(self, city: str) -> Location | None:
        response = requests.get(
            GEOCODE_URL,
            params={"name": city, "count": 1, "language": "en", "format": "json"},
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()
        results = data.get("results") or []
        if not results:
            return None
        first = results[0]
        return Location(
            name=first.get("name", city),
            country=first.get("country", ""),
            latitude=float(first["latitude"]),
            longitude=float(first["longitude"]),
            timezone=first.get("timezone", "auto"),
        )

    def _fetch_current_weather(self, location: Location) -> dict[str, Any]:
        response = requests.get(
            FORECAST_URL,
            params={
                "latitude": location.latitude,
                "longitude": location.longitude,
                "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
                "timezone": location.timezone or "auto",
            },
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()
        return data["current"]

    def _format_weather(self, location: Location, current: dict[str, Any]) -> str:
        code = int(current.get("weather_code", -1))
        description = WEATHER_CODE_DESCRIPTIONS.get(code, f"weather code {code}")
        place = f"{location.name}, {location.country}" if location.country else location.name
        temp = current.get("temperature_2m")
        humidity = current.get("relative_humidity_2m")
        wind = current.get("wind_speed_10m")
        observed_time = current.get("time", "the latest available observation")
        return (
            f"For **{place}**, the latest Open-Meteo reading says it is **{temp}°C** with **{description}**. "
            f"Humidity is about **{humidity}%**, and wind speed is around **{wind} km/h**. "
            f"Observation time: `{observed_time}`."
        )
