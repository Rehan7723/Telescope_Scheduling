# init_db.py
import sqlite3

def initialize_database():
    conn = sqlite3.connect("telescope_schedule.db", check_same_thread=False)
    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS observations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        target TEXT,
        coordinates TEXT,
        duration INTEGER,
        start_time TEXT,
        end_time TEXT,
        priority TEXT,
        wavelength TEXT,
        status TEXT,
        telescope TEXT,
        submitted_at TEXT
    )''')

    c.execute('''CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        telescope TEXT,
        observation_id INTEGER,
        target TEXT,
        completed_at TEXT,
        success INTEGER,
        duration INTEGER,
        priority TEXT
    )''')

    c.execute('''CREATE TABLE IF NOT EXISTS weather_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        lat REAL,
        lon REAL,
        cloud_cover REAL,
        wind_speed REAL,
        humidity REAL
    )''')

    conn.commit()
    conn.close()

if __name__ == "__main__":
    initialize_database()
    print("✅ Database initialized.")
# This script initializes the SQLite database for the telescope scheduling application.
# It creates the necessary tables if they do not already exist.
# Run this script once to set up the database schema.
# Make sure to run this script before using the application to ensure the database is ready.