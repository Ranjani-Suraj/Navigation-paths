import googlemaps
import os
from dotenv import load_dotenv
from pathlib import Path
import asyncio
import logging

from aiohttp import ClientError, ClientSession

from accuweather import (
    AccuWeather,
    ApiError,
    InvalidApiKeyError,
    InvalidCoordinatesError,
    RequestsExceededError,
)


def get_gmaps_client():
    path = Path("Env/.env")
    load_dotenv(dotenv_path=path)
    return googlemaps.Client(key=os.getenv('GMAPS_API_KEY'))

#

async def get_weather_details(LATITUDE, LONGITUDE):
    """Run main function."""
    # LATITUDE = 40.7934
    # LONGITUDE = 77.8600
    API_KEY = "rtfQzLs6SXPqpOiKTFwWZTEOAEBVFsfv"

    logging.basicConfig(level=logging.DEBUG)
    async with ClientSession() as websession:
        try:
            accuweather = AccuWeather(
                API_KEY,
                websession,
                latitude=LATITUDE,
                longitude=LONGITUDE,
                language="pl",
            )
            current_conditions = await accuweather.async_get_current_conditions()
            # forecast_daily = await accuweather.async_get_daily_forecast(
            #     days=5, metric=True
            # )
            # forecast_hourly = await accuweather.async_get_hourly_forecast(
            #     hours=12, metric=True
            # )
        except (
            ApiError,
            InvalidApiKeyError,
            InvalidCoordinatesError,
            ClientError,
            RequestsExceededError,
        ) as error:
            print(f"Error: {error}")
        else:
            # print(f"Location: {accuweather.location_name} ({accuweather.location_key})")
            print(f"Requests remaining: {accuweather.requests_remaining}")
            print(f"Current: {current_conditions}")
            # # print(f"Forecast: {forecast_daily}")
            # print(f"Forecast hourly: {forecast_hourly}")
            raining = current_conditions['PrecipitationSummary']['Precipitation']['Metric']['Value']
            windy = current_conditions['Wind']['Speed']['Imperial']['Value']
            return [raining, windy]



# loop = asyncio.new_event_loop()
# print(loop.run_until_complete(get_weather_details(40.7934, 77.8600)))
# loop.close()

# get_weather_details(40.7934, 77.8600)