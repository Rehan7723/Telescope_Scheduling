# AI-Optimized Telescope Scheduling System

## 📦 Prerequisites

Make sure you have Python 3.10+ installed.

Install required dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Steps to Run the Application

### 1. Initialize the Database
Create the SQLite database and necessary tables:

```bash
python init_db.py
```

### 2. (Optional) Inject Sample Observations
If you'd like to preload the system with real astronomical observation targets:

```bash
python inject_observation.py
```

### 3. Launch the Main Application
Run the main GUI interface:

```bash
python main.py
```

---

## 📂 Files

- `init_db.py` — Creates the database schema
- `inject_observation.py` — Inserts sample observation entries
- `main.py` — Starts the main scheduling GUI with integrated AI logic
- `telescope_schedule.db` — The SQLite database file
- `requirements.txt` — List of Python dependencies

---

## 🧠 Features
- Intelligent scheduling using rule-based and reinforcement learning algorithms
- Real-time weather and visibility checks
- Interactive GUI to manage and monitor telescopes
- Timeline visualization of active and scheduled observations
- Persistent history and logs

---

For issues or contributions, feel free to open an issue or submit a pull request. Happy observing! 🔭

