import sqlite3
from datetime import datetime, timezone

DB_NAME = "telescope_schedule.db"


class Database:
    def __init__(self, db_path=DB_NAME):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
        self.create_tables()  # Ensure tables are created

    def execute(self, query, params=()):
        self.cursor.execute(query, params)
        self.conn.commit()
        return self.cursor

    def fetchall(self, query, params=()):
        self.cursor.execute(query, params)
        return self.cursor.fetchall()

    def fetchone(self, query, params=()):
        self.cursor.execute(query, params)
        return self.cursor.fetchone()

    def close(self):
        self.conn.close()

    def insert_observation(self, obs):
        self.execute(
            """
        INSERT INTO observations (target, coordinates, duration, start_time, end_time, priority, wavelength, status, telescope, submitted_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                obs["target"],
                obs["coordinates"],
                obs["duration"],
                obs["start_time"].isoformat(),
                obs["end_time"].isoformat(),
                obs["priority"],
                obs["wavelength"],
                obs["status"],
                obs.get("telescope"),
                datetime.now(timezone.utc).isoformat(),
            ),
        )

    def update_observation_status(self, obs_id, status, telescope=None):
        self.execute(
            """
        UPDATE observations SET status=?, telescope=? WHERE id=?
        """,
            (status, telescope, obs_id),
        )

    def insert_history_entry(self, obs):
        self.execute(
            """
        INSERT INTO history (telescope, observation_id, target, completed_at, success, duration, priority)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                obs["telescope"],
                obs["id"],
                obs["target"],
                datetime.now(timezone.utc).isoformat(),
                obs["success"],
                obs["duration"],
                obs["priority"],
            ),
        )

    def log_weather(self, telescope_name, lat, lon, cloud, wind, humidity):
        self.execute(
            """
        INSERT INTO weather_log (timestamp, lat, lon, cloud_cover, wind_speed, humidity)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
            (datetime.now(timezone.utc).isoformat(), lat, lon, cloud, wind, humidity),
        )

    def get_pending_observations(self):
        now = datetime.now(timezone.utc).isoformat()
        return self.fetchall(
            """
        SELECT * FROM observations
        WHERE status='Pending' AND start_time <= ? AND end_time >= ?
        """,
            (now, now),
        )

    def get_all_observations(self):
        return self.fetchall("SELECT * FROM observations", ())

    def get_observation_by_id(self, obs_id):
        return self.fetchone("SELECT * FROM observations WHERE id=?", (obs_id,))

    def get_history_entries(self):
        return self.fetchall("SELECT * FROM history ORDER BY completed_at DESC", ())

    def delete_observation(self, obs_id):
        self.execute("DELETE FROM observations WHERE id = ?", (obs_id,))

    def delete_all_observation(self):
        self.execute("DELETE FROM observations", ())

    def get_telescope_stats(self):
        """Return a list of dicts with telescope stats: name, success_count, failure_count, total_observation_time"""
        rows = self.query(
            "SELECT name, success_count, failure_count, total_observation_time FROM telescopes"
        )
        return [dict(row) for row in rows]

    def query(self, sql, params=None):
        """Execute a SELECT query and return rows as a list of dictionaries."""
        self.cursor.execute(sql, params or [])
        columns = [desc[0] for desc in self.cursor.description]
        return [dict(zip(columns, row)) for row in self.cursor.fetchall()]

    def create_tables(self):
        """Create necessary tables if they don't exist."""
        self.cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS telescopes (
            name TEXT PRIMARY KEY,
            success_count INTEGER DEFAULT 0,
            failure_count INTEGER DEFAULT 0,
            total_observation_time INTEGER DEFAULT 0
        )
        """
        )
        self.conn.commit()
