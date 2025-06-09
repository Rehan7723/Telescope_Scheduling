from db import Database
from datetime import datetime, timedelta, timezone

db = Database()

db.delete_all_observation()

print("✅ Sample observations Deleted.")

db.close()
