from datetime import datetime, timezone
import os
from dotenv import load_dotenv
import requests
from timezonefinder import TimezoneFinder
from astral import LocationInfo
from astral.sun import sun
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
import astropy.units as u
import pytz


class RLSchedulingWrappert:
    def __init__(self, model, telescopes, observations, schedule, log_fn=print):
        self.model = model
        self.telescopes = telescopes
        self.observations = observations
        self.schedule = schedule
        self.log = log_fn

    def fetch_weather_for_location(self, lat, lon):
        load_dotenv()
        API_KEY = os.getenv("API_KEY")
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
        try:
            data = requests.get(url).json()
            cloud = data["clouds"]["all"] / 100.0
            self.log(f"☁️  Cloud cover at ({lat}, {lon}): {cloud}")
            return cloud
        except Exception as e:
            self.log(f"⚠️  Weather fetch failed at ({lat}, {lon}): {e}")
            return 0.0
    
    def fetchD_weather_for_location(self, lat, lon):
        """
        Dummy version for testing without external API.
        Simulates weather data but maintains structure.
        """
        import random
        try:
            cloud = round(random.uniform(0.0, 1.0), 2) # Simulate cloud cover between 0.0 and 1.0
            self.log(f"☁️ Cloud cover at ({lat}, {lon}): {cloud}")
            return cloud
        except Exception as e:
            self.log(f"⚠️ Weather fetch failed at ({lat}, {lon}): {e}")
            return 0.0

    def is_target_visible(self, ra_dec_str, obs_time, lat, lon):
        try:
            coord = SkyCoord(ra_dec_str, unit=(u.hourangle, u.deg))
            loc = EarthLocation(lat=lat * u.deg, lon=lon * u.deg)
            time = Time(obs_time)
            altaz = coord.transform_to(AltAz(obstime=time, location=loc))
            visible = altaz.alt.deg > 20
            self.log(f"🔭 Target visibility at {lat}, {lon}: {altaz.alt.deg:.2f}° → {'YES' if visible else 'NO'}")
            return visible
        except Exception as e:
            self.log(f"⚠️  Visibility check failed for {ra_dec_str}: {e}")
            return False

    def is_night(self, utc_time, lat, lon):
        try:
            tf = TimezoneFinder()
            tz = pytz.timezone(tf.timezone_at(lat=lat, lng=lon))
            local_time = utc_time.astimezone(tz)
            loc = LocationInfo(latitude=lat, longitude=lon)
            s = sun(loc.observer, date=local_time.date(), tzinfo=local_time.tzinfo)
            night = local_time < s["sunrise"] or local_time > s["sunset"]
            self.log(f"🌙 Night check at ({lat}, {lon}) → {night} (Local time: {local_time.time()})")
            return night
        except Exception as e:
            self.log(f"⚠️  Night check failed: {e}")
            return False

    def run_step(self):
        now = datetime.utcnow().replace(tzinfo=timezone.utc)
        available_telescopes = [
            t for t in self.telescopes if t["status"] == "Operational" and not t["current_observation"]
        ]
        pending_obs = []
        for o in self.observations:
            start_time = o["start_time"]
            end_time = o["end_time"]
            if isinstance(start_time, str):
                try:
                    start_time = datetime.fromisoformat(start_time)
                except ValueError:
                    start_time = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S%z")
            if isinstance(end_time, str):
                try:
                    end_time = datetime.fromisoformat(end_time)
                except ValueError:
                    end_time = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S%z")
            if o["status"] == "Pending" and start_time <= now <= end_time:
                pending_obs.append(o)

        if not pending_obs or not available_telescopes:
            print("🚫 No observations or telescopes to schedule.")
            return

        cloud_cover = [
            self.fetchD_weather_for_location(t["lat"], t["lon"]) for t in available_telescopes
        ]

        for obs in pending_obs:
            self.log(f"\n🔍 Evaluating observation: {obs['target']} ({obs['coordinates']})")

            for i, telescope in enumerate(available_telescopes):
                self.log(f"➡️ Checking telescope: {telescope['name']}")

                if obs["wavelength"] not in telescope["capabilities"]:
                    self.log(f"   ❌ Capability mismatch: {obs['wavelength']} not in {telescope['capabilities']}")
                    continue

                if obs["wavelength"] not in ["Radio"] and not self.is_night(now, telescope["lat"], telescope["lon"]):
                    self.log("   🌞 Not nighttime at telescope location.")
                    continue

                if not self.is_target_visible(obs["coordinates"], now, telescope["lat"], telescope["lon"]):
                    self.log("   🚫 Target not visible (below 20° altitude).")
                    continue

                if obs["wavelength"] in ["Optical", "UV"] and cloud_cover[i] > 0.7:
                    self.log(f"   ☁️ Too cloudy for optical/UV ({cloud_cover[i]:.2f})")
                    continue

                # Passed all checks
                self.log(f"✅ Scheduling {obs['target']} on {telescope['name']}", tag="green")
                obs["status"] = "Scheduled"
                obs["telescope"] = telescope["name"]
                telescope["current_observation"] = obs
                self.schedule.append(obs)
                break  # Only schedule one per step