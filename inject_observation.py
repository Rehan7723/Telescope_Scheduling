# inject_observations.py

from db import Database
from datetime import datetime, timedelta, timezone

db = Database()

now = datetime.now(timezone.utc)

sample_observations = [
    {
        "target": "Andromeda Galaxy",
        "coordinates": "00h42m44.3s +41d16m9s",
        "duration": 30,
        "start_time": now,
        "end_time": now + timedelta(hours=1),
        "priority": "High",
        "wavelength": "Radio",
        "status": "Pending",
        "telescope": None,
    },
    {
        "target": "Crab Nebula",
        "coordinates": "05h34m31.94s +22d00m52.2s",
        "duration": 20,
        "start_time": now,
        "end_time": now + timedelta(hours=2),
        "priority": "Medium",
        "wavelength": "Infrared",
        "status": "Pending",
        "telescope": None,
    },
    {
        "target": "Centaurus A",
        "coordinates": "13h25m27.6s -43d01m09s",
        "duration": 40,
        "start_time": now - timedelta(minutes=15),
        "end_time": now + timedelta(hours=1),
        "priority": "Critical",
        "wavelength": "Radio",
        "status": "Pending",
        "telescope": None,
    },
    {
        "target": "Orion Nebula",
        "coordinates": "05h35m17.3s -05d23m28s",
        "duration": 25,
        "start_time": now,
        "end_time": now + timedelta(hours=1),
        "priority": "Low",
        "wavelength": "Radio",
        "status": "Pending",
        "telescope": None,
    },
    {
        "target": "Sirius",
        "coordinates": "06h45m08.9s -16d42m58s",
        "duration": 20,
        "start_time": now,
        "end_time": now + timedelta(hours=1),
        "priority": "Medium",
        "wavelength": "Optical",
        "status": "Pending",
        "telescope": None,
    },
    {
        "target": "Vega",
        "coordinates": "18h36m56.3s +38d47m01s",
        "duration": 30,
        "start_time": now,
        "end_time": now + timedelta(hours=2),
        "priority": "High",
        "wavelength": "Infrared",
        "status": "Pending",
        "telescope": None,
    },
    {
        "target": "Messier 87 (M87)",
        "coordinates": "12h30m49.4s +12d23m28s",
        "duration": 45,
        "start_time": now,
        "end_time": now + timedelta(hours=3),
        "priority": "Critical",
        "wavelength": "Radio",
        "status": "Pending",
        "telescope": None,
    },
    {
        "target": "Polaris",
        "coordinates": "02h31m49.09s +89d15m50.8s",
        "duration": 15,
        "start_time": now,
        "end_time": now + timedelta(hours=3),
        "priority": "Low",
        "wavelength": "Radio",
        "status": "Pending",
        "telescope": None,
    },
    {
        "target": "Sagittarius A*",
        "coordinates": "17h45m40.04s -29d00m28.1s",
        "duration": 60,
        "start_time": now,
        "end_time": now + timedelta(hours=4),
        "priority": "Critical",
        "wavelength": "Radio",
        "status": "Pending",
        "telescope": None,
    },
    {
        "target": "Large Magellanic Cloud",
        "coordinates": "05h23m34.6s -69d45m22s",
        "duration": 50,
        "start_time": now,
        "end_time": now + timedelta(hours=8),
        "priority": "High",
        "wavelength": "Optical",
        "status": "Pending",
        "telescope": None,
    },
    {
        "target": "Antares",
        "coordinates": "16h29m24.4s -26d25m55s",
        "duration": 35,
        "start_time": now,
        "end_time": now + timedelta(hours=6),
        "priority": "Medium",
        "wavelength": "Radio",
        "status": "Pending",
        "telescope": None,
    },
]

for obs in sample_observations:
    db.insert_observation(obs)

print("✅ Sample observations injected.")

db.close()
