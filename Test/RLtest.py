import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime, timedelta , timezone
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import time
import threading
from collections import defaultdict
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

# RL Model Integration
from stable_baselines3 import PPO
import numpy as np


class RLSchedulingWrapper:
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
        pending_obs = [
            o for o in self.observations if o["status"] == "Pending" and o["start_time"] <= now <= o["end_time"]
        ]

        if not pending_obs or not available_telescopes:
            # print("🚫 No observations or telescopes to schedule.")
            return

        cloud_cover = [
            self.fetch_weather_for_location(t["lat"], t["lon"]) for t in available_telescopes
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



class RealTimeTelescopeScheduler:
    def __init__(self, root):
        self.root = root
        self.root.title("AI-Optimized Telescope Scheduler - Real-Time")

        # Initialize system state with more detailed data
        load_dotenv()
        self.observations = []
        self.telescopes = [
            {
                "name": "Very Large Telescope",
                "lat": -24.6,
                "lon": -70.4,
                "capabilities": ["Optical", "Infrared","Radio"],
                "status": "Operational",
                "current_observation": None,
            },
            {
                "name": "Keck Observatory",
                "lat": 19.8,
                "lon": -155.5,
                "capabilities": ["Optical", "Infrared", "Radio"],
                "status": "Operational",
                "current_observation": None,
            },
            {
                "name": "Gran Telescopio Canarias",
                "lat": 28.8,
                "lon": -17.9,
                "capabilities": ["Optical", "Radio"],
                "status": "Operational",
                "current_observation": None,
            },
        ]

        self.schedule = []
        self.history = []
        self.last_update = datetime.now(timezone.utc)
        self.weather_data = {
            "cloud_cover": 0.0,
            "timestamp": datetime.now(timezone.utc),
            "wind_speed": random.uniform(0, 20),
            "humidity": random.uniform(0, 100),
        }

        # Create all GUI elements
        self.create_widgets()

        # Initialize RL model if available
        try:
            self.rl_model = PPO.load("ppo_telescope_scheduler.zip")
            self.rl_scheduler = RLSchedulingWrapper(
                self.rl_model, self.telescopes, self.observations, self.schedule, log_fn=self.log
            )
            print("RL model loaded successfully.")
        except Exception as e:
            self.rl_model = None
            self.rl_scheduler = None
            print(f"Failed to load RL model: {e}")

        # Start periodic updates in a separate thread
        self.update_thread = threading.Thread(target=self.update_loop, daemon=True)
        self.update_thread.start()

        # Initialize data displays
        self.update_all_displays()

    def create_widgets(self):
        # Configure main window layout
        self.root.geometry("1200x900")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Create notebook for tabbed interface
        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=0, column=0, sticky="nsew")

        # Create tabs
        self.create_observation_tab()
        self.create_schedule_tab()
        self.create_monitoring_tab()
        self.create_analytics_tab()
        self.create_comparison_tab()

        # Status bar at bottom
        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(
            self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W
        )
        self.status_bar.grid(row=1, column=0, sticky="ew")
        self.update_status("System initialized and ready")

    def create_observation_tab(self):
        """Tab for submitting and managing observation requests"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Observations")

        # Configure grid
        tab.columnconfigure(0, weight=1)
        tab.columnconfigure(1, weight=1)
        tab.rowconfigure(1, weight=1)

        # Submit observation frame
        submit_frame = ttk.LabelFrame(tab, text="Submit New Observation", padding=10)
        submit_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        # Observation form
        ttk.Label(submit_frame, text="Target:").grid(row=0, column=0, sticky="w")
        self.target_entry = ttk.Entry(submit_frame, width=30)
        self.target_entry.grid(row=0, column=1, sticky="ew", pady=2)

        ttk.Label(submit_frame, text="Coordinates (RA/DEC):").grid(
            row=1, column=0, sticky="w"
        )
        self.coord_entry = ttk.Entry(submit_frame, width=30)
        self.coord_entry.grid(row=1, column=1, sticky="ew", pady=2)

        ttk.Label(submit_frame, text="Duration (min):").grid(
            row=2, column=0, sticky="w"
        )
        self.duration_entry = ttk.Entry(submit_frame, width=10)
        self.duration_entry.grid(row=2, column=1, sticky="w", pady=2)

        ttk.Label(submit_frame, text="Start Time:").grid(row=3, column=0, sticky="w")
        self.start_entry = ttk.Entry(submit_frame, width=20)
        self.start_entry.insert(0, datetime.utcnow().replace(tzinfo=timezone.utc).strftime("%Y-%m-%d %H:%M"))
        self.start_entry.grid(row=3, column=1, sticky="w", pady=2)

        ttk.Label(submit_frame, text="End Time:").grid(row=4, column=0, sticky="w")
        self.end_entry = ttk.Entry(submit_frame, width=20)
        self.end_entry.insert(0, (datetime.utcnow().replace(tzinfo=timezone.utc) + timedelta(hours=2)).strftime("%Y-%m-%d %H:%M"))
        self.end_entry.grid(row=4, column=1, sticky="w", pady=2)

        ttk.Label(submit_frame, text="Priority:").grid(row=5, column=0, sticky="w")
        self.priority_combo = ttk.Combobox(
            submit_frame, values=["Low", "Medium", "High", "Critical"]
        )
        self.priority_combo.current(1)
        self.priority_combo.grid(row=5, column=1, sticky="w", pady=2)

        ttk.Label(submit_frame, text="Wavelength:").grid(row=6, column=0, sticky="w")
        self.wavelength_combo = ttk.Combobox(
            submit_frame, values=["Optical", "Infrared", "UV", "X-ray", "Radio"]
        )
        self.wavelength_combo.current(0)
        self.wavelength_combo.grid(row=6, column=1, sticky="w", pady=2)

        ttk.Button(
            submit_frame, text="Submit Request", command=self.submit_observation
        ).grid(row=7, column=0, columnspan=2, pady=5)

        # Observation list frame
        list_frame = ttk.LabelFrame(tab, text="Observation Queue", padding=10)
        list_frame.grid(row=0, column=1, rowspan=2, padx=5, pady=5, sticky="nsew")

        # Treeview for observations
        self.obs_tree = ttk.Treeview(
            list_frame,
            columns=("target", "priority", "status", "telescope"),
            selectmode="browse",
            height=15,
        )
        self.obs_tree.heading("#0", text="ID")
        self.obs_tree.heading("target", text="Target")
        self.obs_tree.heading("priority", text="Priority")
        self.obs_tree.heading("status", text="Status")
        self.obs_tree.heading("telescope", text="Telescope")

        self.obs_tree.column("#0", width=50)
        self.obs_tree.column("target", width=150)
        self.obs_tree.column("priority", width=80)
        self.obs_tree.column("status", width=100)
        self.obs_tree.column("telescope", width=120)

        self.obs_tree.grid(row=0, column=0, sticky="nsew")

        # Scrollbar
        scrollbar = ttk.Scrollbar(
            list_frame, orient="vertical", command=self.obs_tree.yview
        )
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.obs_tree.configure(yscrollcommand=scrollbar.set)

        # Priority update controls
        update_frame = ttk.Frame(list_frame)
        update_frame.grid(row=1, column=0, columnspan=2, pady=5, sticky="ew")

        ttk.Label(update_frame, text="New Priority:").pack(side="left", padx=5)
        self.new_priority_combo = ttk.Combobox(
            update_frame, values=["Low", "Medium", "High", "Critical"]
        )
        self.new_priority_combo.current(1)
        self.new_priority_combo.pack(side="left", padx=5)

        ttk.Button(
            update_frame, text="Update Priority", command=self.update_priority
        ).pack(side="left", padx=5)
        ttk.Button(
            update_frame, text="Delete Request", command=self.delete_observation
        ).pack(side="left", padx=5)

        # Configure weights
        list_frame.rowconfigure(0, weight=1)
        list_frame.columnconfigure(0, weight=1)

    def create_schedule_tab(self):
        """Tab for viewing and managing the schedule"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Schedule")

        # Configure grid
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(0, weight=1)

        # Schedule display frame
        schedule_frame = ttk.LabelFrame(tab, text="Observation Schedule", padding=10)
        schedule_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        # Create a canvas for the timeline visualization
        self.schedule_canvas = tk.Canvas(schedule_frame, bg="white", height=300)
        self.schedule_canvas.grid(row=0, column=0, sticky="nsew", pady=5)

        # Schedule controls
        control_frame = ttk.Frame(schedule_frame)
        control_frame.grid(row=1, column=0, sticky="ew")

        ttk.Button(
            control_frame, text="Generate Schedule", command=self.generate_schedule
        ).pack(side="left", padx=5)
        # ttk.Button(
        #     control_frame, text="Optimize with AI", command=self.optimize_schedule
        # ).pack(side="left", padx=5)
        ttk.Button(
            control_frame, text="Refresh View", command=self.update_schedule_display
        ).pack(side="left", padx=5)

        # Current schedule details
        self.schedule_text = tk.Text(schedule_frame, height=10, wrap=tk.WORD)
        self.schedule_text.grid(row=2, column=0, sticky="nsew", pady=5)

        # Configure weights
        schedule_frame.columnconfigure(0, weight=1)
        schedule_frame.rowconfigure(0, weight=1)
        schedule_frame.rowconfigure(2, weight=1)

    def create_monitoring_tab(self):
        """Tab for monitoring telescope status and weather"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Monitoring")

        # Configure grid
        tab.columnconfigure(0, weight=1)
        tab.columnconfigure(1, weight=1)
        tab.rowconfigure(0, weight=1)

        # Telescope status frame
        telescope_frame = ttk.LabelFrame(tab, text="Telescope Status", padding=10)
        telescope_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        # Treeview for telescope status
        self.telescope_tree = ttk.Treeview(
            telescope_frame,
            columns=("name", "status", "observation", "remaining"),
            height=10,
        )
        self.telescope_tree.heading("#0", text="ID")
        self.telescope_tree.heading("name", text="Telescope")
        self.telescope_tree.heading("status", text="Status")
        self.telescope_tree.heading("observation", text="Observation")
        self.telescope_tree.heading("remaining", text="Time Remaining")

        self.telescope_tree.column("#0", width=50)
        self.telescope_tree.column("name", width=150)
        self.telescope_tree.column("status", width=100)
        self.telescope_tree.column("observation", width=150)
        self.telescope_tree.column("remaining", width=100)

        self.telescope_tree.grid(row=0, column=0, sticky="nsew")

        # Telescope controls
        control_frame = ttk.Frame(telescope_frame)
        control_frame.grid(row=1, column=0, sticky="ew")

        ttk.Button(
            control_frame, text="Maintenance Mode", command=self.set_maintenance_mode
        ).pack(side="left", padx=5)
        ttk.Button(
            control_frame, text="Refresh Status", command=self.update_telescope_display
        ).pack(side="left", padx=5)

        # Weather frame
        Logg_frame = ttk.LabelFrame(tab, text="Weather Conditions", padding=10)
        Logg_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

        # # Weather visualization
        # self.weather_fig = Figure(figsize=(5, 4), dpi=100)
        # self.weather_ax = self.weather_fig.add_subplot(111)
        # self.weather_canvas = FigureCanvasTkAgg(self.weather_fig, master=weather_frame)
        # self.weather_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Log console (Text widget with vertical scrollbar)
        log_frame = ttk.Frame(Logg_frame)
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.log_console = tk.Text(log_frame, wrap=tk.WORD, height=20, state=tk.DISABLED)
        self.log_console_scrollbar = ttk.Scrollbar(log_frame, orient="vertical")
        self.log_console_scrollbar.config(command=self.log_console.yview)
        self.log_console.configure(yscrollcommand=self._on_log_scroll)


        self.log_console.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.log_console_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._log_at_bottom = True


        # Weather controls
        # ttk.Button(
        #     weather_frame, text="Refresh Weather", command=self.fetch_weather
        # ).pack(side=tk.BOTTOM, pady=5)

        # Configure weights
        telescope_frame.columnconfigure(0, weight=1)
        telescope_frame.rowconfigure(0, weight=1)
        Logg_frame.columnconfigure(0, weight=1)
        Logg_frame.rowconfigure(0, weight=1)

    def create_analytics_tab(self):
        """Tab for viewing analytics and history"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Analytics")

        # Configure grid
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(0, weight=1)

        # Analytics frame
        analytics_frame = ttk.LabelFrame(tab, text="Observation Analytics", padding=10)
        analytics_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        # Create notebook within analytics tab
        analytics_notebook = ttk.Notebook(analytics_frame)
        analytics_notebook.grid(row=0, column=0, sticky="nsew")

        # History tab
        history_tab = ttk.Frame(analytics_notebook)
        analytics_notebook.add(history_tab, text="History")

        self.history_text = tk.Text(history_tab, wrap=tk.WORD)
        self.history_text.pack(expand=True, fill="both")

        # Statistics tab
        stats_tab = ttk.Frame(analytics_notebook)
        analytics_notebook.add(stats_tab, text="Statistics")

        # Create figure for statistics
        self.stats_fig = Figure(figsize=(5, 4), dpi=100)
        self.stats_ax = self.stats_fig.add_subplot(111)
        self.stats_canvas = FigureCanvasTkAgg(self.stats_fig, master=stats_tab)
        self.stats_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Configure weights
        analytics_frame.columnconfigure(0, weight=1)
        analytics_frame.rowconfigure(0, weight=1)

    def create_comparison_tab(self):
        """New tab for comparing telescope performance"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Comparison")

        # Configure grid
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(0, weight=1)

        # Comparison frame
        comparison_frame = ttk.LabelFrame(
            tab, text="Telescope Performance Comparison", padding=10
        )
        comparison_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        # Create figure for comparison
        self.comp_fig = Figure(figsize=(8, 6), dpi=100)
        self.comp_ax1 = self.comp_fig.add_subplot(211)
        self.comp_ax2 = self.comp_fig.add_subplot(212)

        self.comp_canvas = FigureCanvasTkAgg(self.comp_fig, master=comparison_frame)
        self.comp_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Refresh button
        ttk.Button(
            comparison_frame,
            text="Refresh Comparison",
            command=self.update_comparison_display,
        ).pack(side=tk.BOTTOM, pady=5)

        # Configure weights
        comparison_frame.columnconfigure(0, weight=1)
        comparison_frame.rowconfigure(0, weight=1)

    def update_loop(self):
        """Main update loop running in background thread"""
        while True:
            try:
                # # Update weather periodically
                # if (
                #     datetime.now(timezone.utc) - self.weather_data["timestamp"]
                # ).seconds > 300:  # 5 minutes
                #     self.fetch_weather()

                # Update system status
                self.update_system_status()

                # Update all displays in the main thread
                self.root.after(0, self.update_all_displays)

                # Sleep for a short interval (10 seconds)
                time.sleep(10)

            except Exception as e:
                self.update_status(f"Error in update loop: {str(e)}")
                time.sleep(5)

    def update_all_displays(self):
        """Update all GUI displays"""
        self.update_observation_list()
        self.update_telescope_display()
        self.update_schedule_display()
        self.update_history_display()
        # self.update_weather_display()
        self.update_stats_display()
        self.update_comparison_display()
        self.update_status(
            f"Last update: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}"
        )

    def update_status(self, message):
        """Update the status bar"""
        self.status_var.set(message)

    def update_system_status(self):
        """Update the system state"""
        now = datetime.now(timezone.utc)

        # Check for completed observations
        for telescope in self.telescopes:
            if telescope["current_observation"]:
                obs = telescope["current_observation"]
                if "end_time" in obs and obs["end_time"] <= now:
                    # Determine if observation was successful (90% chance)
                    success = random.random() > 0.1

                    # Update telescope statistics
                    if success:
                        telescope["success_count"] += 1
                    else:
                        telescope["failure_count"] += 1

                    telescope["total_observation_time"] += obs["duration"]

                    # Move to history
                    self.history.append(
                        {
                            "telescope": telescope["name"],
                            "observation": obs,
                            "completed_at": now,
                            "success": success,
                        }
                    )
                    telescope["current_observation"] = None
                    telescope["status"] = "Operational"

        # Assign new observations to all available telescopes simultaneously
        # Assign new observations using RL if model is available
        if self.rl_scheduler:
            self.rl_scheduler.run_step()
        else:
            # Fallback to rule-based logic if RL not available
            available_telescopes = [
                t
                for t in self.telescopes
                if t["status"] == "Operational" and not t["current_observation"]
            ]

            priority_order = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}

            pending_obs = [
                obs
                for obs in self.observations
                if obs["status"] == "Pending"
                and obs["start_time"] <= now <= obs["end_time"]
            ]

            if pending_obs:
                pending_obs.sort(
                    key=lambda x: (-priority_order[x["priority"]], x["end_time"])
                )

                for telescope in available_telescopes:
                    for obs in pending_obs:
                        if (
                            obs["status"] == "Pending"
                            and obs["wavelength"] in telescope["capabilities"]
                        ):
                            if (
                                obs["wavelength"] in ["Optical", "UV"]
                                and self.weather_data["cloud_cover"] > 0.7
                            ):
                                continue
                            obs["status"] = "Scheduled"
                            obs["telescope"] = telescope["name"]
                            self.schedule.append(obs)
                            break

        # Process scheduled observations to start them immediately if telescope is available
        for obs in self.schedule[:]:  # Iterate over a copy
            if obs["status"] == "Scheduled" and obs["telescope"]:
                telescope = next(
                    t for t in self.telescopes if t["name"] == obs["telescope"]
                )

                if (
                    telescope["status"] == "Operational"
                    and not telescope["current_observation"]
                ):
                    telescope["current_observation"] = obs
                    telescope["status"] = "Observing"
                    obs["status"] = "In Progress"
                    obs["start_time_actual"] = now
                    obs["end_time"] = now + timedelta(minutes=obs["duration"])
                    self.schedule.remove(obs)

        # Remove completed observations from schedule
        self.schedule = [obs for obs in self.schedule if obs["status"] != "In Progress"]

    def submit_observation(self):
        """Submit a new observation request"""
        try:
            target = self.target_entry.get()
            coordinates = self.coord_entry.get()
            duration = int(self.duration_entry.get())
            start_time_input = datetime.strptime(self.start_entry.get(), "%Y-%m-%d %H:%M")
            now = datetime.utcnow().replace(tzinfo=timezone.utc)
            start_time = max(start_time_input.replace(tzinfo=timezone.utc), now)
            # Handle end time (based on duration or field input)
            if self.end_entry.get().strip():
                end_time_input = datetime.strptime(self.end_entry.get(), "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
                end_time = max(end_time_input, start_time + timedelta(minutes=duration))
            else:
                end_time = start_time + timedelta(minutes=duration)
            priority = self.priority_combo.get()
            wavelength = self.wavelength_combo.get()

            if end_time <= start_time:
                raise ValueError("End time must be after start time")
            if duration <= 0:
                raise ValueError("Duration must be positive")
            if not target or not coordinates:
                raise ValueError("Target and coordinates are required")

            self.observations.append(
                {
                    "id": len(self.observations) + 1,
                    "target": target,
                    "coordinates": coordinates,
                    "duration": duration,
                    "start_time": start_time,
                    "end_time": end_time,
                    "priority": priority,
                    "wavelength": wavelength,
                    "status": "Pending",
                    "telescope": None,
                    "submitted_at": datetime.now(timezone.utc)
                }
            )

            messagebox.showinfo(
                "Success", f"Observation '{target}' submitted successfully"
            )
            self.update_observation_list()

        except ValueError as e:
            messagebox.showerror("Error", f"Invalid input: {str(e)}")

    def update_observation_list(self):
        """Update the observation list display"""
        # Preserve current selection
        selected = self.obs_tree.selection()
        selected_id = None
        if selected:
            selected_id = self.obs_tree.item(selected[0], "text")

        self.obs_tree.delete(*self.obs_tree.get_children())

        for obs in sorted(
            self.observations, key=lambda x: (x["priority"], x["start_time"])
        ):
            item = self.obs_tree.insert(
                "",
                "end",
                text=obs["id"],
                values=(
                    obs["target"],
                    obs["priority"],
                    obs["status"],
                    obs.get("telescope", ""),
                ),
            )
            # Restore selection if this is the previously selected item
            if selected_id is not None and str(obs["id"]) == str(selected_id):
                self.obs_tree.selection_set(item)

    def update_priority(self):
        """Update observation priority"""
        selected = self.obs_tree.selection()
        if not selected:
            messagebox.showwarning("Warning", "Please select an observation first")
            return

        new_priority = self.new_priority_combo.get()
        obs_id = int(self.obs_tree.item(selected[0], "text"))

        for obs in self.observations:
            if obs["id"] == obs_id:
                obs["priority"] = new_priority
                break

        messagebox.showinfo("Success", "Priority updated successfully")
        self.update_observation_list()

    def delete_observation(self):
        """Delete an observation request"""
        selected = self.obs_tree.selection()
        if not selected:
            messagebox.showwarning("Warning", "Please select an observation first")
            return

        obs_id = int(self.obs_tree.item(selected[0], "text"))

        # Remove from observations list
        self.observations = [obs for obs in self.observations if obs["id"] != obs_id]

        # Remove from schedule if it's there
        self.schedule = [obs for obs in self.schedule if obs["id"] != obs_id]

        messagebox.showinfo("Success", "Observation deleted successfully")
        self.update_observation_list()
        self.update_schedule_display()

    def update_telescope_display(self):
        """Update the telescope status display"""
        self.telescope_tree.delete(*self.telescope_tree.get_children())

        for i, telescope in enumerate(self.telescopes, 1):
            if telescope["current_observation"]:
                obs = telescope["current_observation"]
                remaining = (obs["end_time"] - datetime.now(timezone.utc)).seconds // 60
                remaining_str = f"{remaining} min" if remaining > 0 else "Complete"
                values = (
                    telescope["name"],
                    telescope["status"],
                    obs["target"],
                    remaining_str,
                )
            else:
                values = (telescope["name"], telescope["status"], "None", "N/A")

            self.telescope_tree.insert("", "end", text=i, values=values)

    def set_maintenance_mode(self):
        """Put selected telescope in maintenance mode"""
        selected = self.telescope_tree.selection()
        if not selected:
            messagebox.showwarning("Warning", "Please select a telescope first")
            return

        telescope_idx = int(self.telescope_tree.item(selected[0], "text")) - 1
        telescope = self.telescopes[telescope_idx]

        if telescope["status"] == "Maintenance":
            telescope["status"] = "Operational"
            messagebox.showinfo(
                "Info", f"{telescope['name']} returned to operational status"
            )
        else:
            telescope["status"] = "Maintenance"
            if telescope["current_observation"]:
                # Reschedule the interrupted observation
                obs = telescope["current_observation"]
                obs["status"] = "Pending"
                self.schedule.append(obs)
                telescope["current_observation"] = None

            messagebox.showinfo("Info", f"{telescope['name']} set to maintenance mode")

        self.update_telescope_display()
        self.update_schedule_display()

    def generate_schedule(self):
        """Generate an initial observation schedule"""
        if not self.observations:
            messagebox.showwarning("Warning", "No observations to schedule")
            return

        # Clear current schedule (except in-progress observations)
        self.schedule = [obs for obs in self.schedule if obs["status"] == "In Progress"]

        # Get all pending observations within their time windows
        now = datetime.now(timezone.utc)
        pending_obs = [
            obs
            for obs in self.observations
            if obs["status"] == "Pending"
            and obs["start_time"] <= now <= obs["end_time"]
        ]

        if not pending_obs:
            messagebox.showinfo(
                "Info", "No pending observations within their time windows"
            )
            return

        # Sort by priority (Critical first) and then by earliest deadline
        priority_order = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}
        pending_obs.sort(key=lambda x: (-priority_order[x["priority"]], x["end_time"]))

        # Assign observations to all available telescopes simultaneously
        available_telescopes = [
            t
            for t in self.telescopes
            if t["status"] == "Operational" and not t["current_observation"]
        ]

        for telescope in available_telescopes:
            for obs in pending_obs:
                if (
                    obs["status"] == "Pending"
                    and obs["wavelength"] in telescope["capabilities"]
                ):

                    # Check weather conditions for optical/UV observations
                    if (
                        obs["wavelength"] in ["Optical", "UV"]
                        and self.weather_data["cloud_cover"] > 0.7
                    ):
                        continue  # Skip if too cloudy

                    # Add to schedule
                    obs["status"] = "Scheduled"
                    obs["telescope"] = telescope["name"]
                    self.schedule.append(obs)
                    break  # Move to next telescope

        messagebox.showinfo(
            "Success",
            f"Schedule generated with {len([o for o in self.schedule if o['status'] == 'Scheduled'])} new observations",
        )
        self.update_schedule_display()

    # def optimize_schedule(self):
        """AI-based schedule optimization"""
        if not self.schedule:
            messagebox.showwarning("Warning", "Please generate a schedule first")
            return

        original_count = len(self.schedule)
        now = datetime.now(timezone.utc)

        # Simulate AI optimization with more sophisticated logic
        # 1. Try to schedule more observations if conditions allow
        if self.weather_data["cloud_cover"] < 0.5:
            for obs in self.observations:
                if (
                    obs["status"] == "Pending"
                    and obs["start_time"] <= now <= obs["end_time"]
                    and len(self.schedule) < len(self.telescopes) * 3
                ):  # Don't overload schedule

                    # Find suitable telescope
                    for telescope in self.telescopes:
                        if (
                            telescope["status"] == "Operational"
                            and not telescope["current_observation"]
                            and obs["wavelength"] in telescope["capabilities"]
                        ):

                            # Add to schedule
                            obs["status"] = "Scheduled"
                            obs["telescope"] = telescope["name"]
                            self.schedule.append(obs)
                            break

        # 2. Re-prioritize based on remaining time and priority
        priority_order = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}
        self.schedule.sort(
            key=lambda x: (
                -priority_order[x["priority"]],  # Higher priority first
                x["end_time"],  # Earlier deadlines first
                -x[
                    "duration"
                ],  # Longer observations first (more likely to miss window)
            )
        )

        # 3. Adjust for telescope efficiency
        for obs in self.schedule:
            if obs["status"] == "Scheduled" and obs["telescope"]:
                telescope = next(
                    t for t in self.telescopes if t["name"] == obs["telescope"]
                )
                # Adjust duration based on telescope efficiency
                obs["duration"] = int(obs["duration"] / telescope["efficiency"])

        messagebox.showinfo(
            "AI Optimization",
            f"AI improved schedule from {original_count} to {len(self.schedule)} observations",
        )
        self.update_schedule_display()

    def update_schedule_display(self):
        """Update the schedule display with text and visualization"""
        # Update text display
        self.schedule_text.delete(1.0, tk.END)

        if not self.schedule and not any(
            t["current_observation"] for t in self.telescopes
        ):
            self.schedule_text.insert(tk.END, "No observations scheduled yet\n")
            return

        self.schedule_text.insert(tk.END, "=== Current Observation Schedule ===\n")
        self.schedule_text.insert(
            tk.END, f"Generated at: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}\n"
        )
        self.schedule_text.insert(
            tk.END,
            f"Weather Conditions: {self.weather_data['cloud_cover']:.0%} cloud cover\n\n",
        )

        # Show currently running observations first
        running_obs = []
        for telescope in self.telescopes:
            if telescope["current_observation"]:
                running_obs.append(telescope["current_observation"])

        if running_obs:
            self.schedule_text.insert(tk.END, "=== Currently Observing ===\n")
            for i, obs in enumerate(running_obs, 1):
                remaining = (obs["end_time"] - datetime.now(timezone.utc)).seconds // 60
                self.schedule_text.insert(
                    tk.END,
                    f"{i}. {obs['target']} ({obs['duration']} min)\n"
                    f"   Telescope: {obs['telescope']}\n"
                    f"   Coordinates: {obs['coordinates']}\n"
                    f"   Priority: {obs['priority']}\n"
                    f"   Wavelength: {obs['wavelength']}\n"
                    f"   Time Remaining: {remaining} minutes\n\n",
                )

        if self.schedule:
            self.schedule_text.insert(tk.END, "=== Scheduled Observations ===\n")
            for i, obs in enumerate(self.schedule, len(running_obs) + 1):
                if obs["status"] == "Scheduled":
                    self.schedule_text.insert(
                        tk.END,
                        f"{i}. {obs['target']} ({obs['duration']} min)\n"
                        f"   Telescope: {obs.get('telescope', 'Not assigned')}\n"
                        f"   Coordinates: {obs['coordinates']}\n"
                        f"   Priority: {obs['priority']}\n"
                        f"   Wavelength: {obs['wavelength']}\n"
                        f"   Window: {obs['start_time'].strftime('%Y-%m-%d %H:%M')} to "
                        f"{obs['end_time'].strftime('%Y-%m-%d %H:%M')}\n\n",
                    )

        # Update schedule visualization
        self.draw_schedule_timeline()

    def draw_schedule_timeline(self):
        """Draw a timeline visualization of the schedule"""
        self.schedule_canvas.delete("all")

        now = datetime.now(timezone.utc)
        start_time = now - timedelta(hours=1)
        end_time = now + timedelta(hours=6)
        total_seconds = (end_time - start_time).total_seconds()

        # Layout constants
        label_margin = 155  # space for telescope names on the left
        right_margin = 70
        canvas_width = self.schedule_canvas.winfo_width()
        canvas_height = self.schedule_canvas.winfo_height()
        timeline_width = canvas_width - label_margin - right_margin

        # Draw timeline axis
        self.schedule_canvas.create_line(label_margin, 30, canvas_width - right_margin, 30, width=2)

        # Draw time markers (every hour)
        for i in range(7):
            time_pos = start_time + timedelta(hours=i)
            x = label_margin + (i * timeline_width / 6)
            self.schedule_canvas.create_line(x, 25, x, 35, width=1)
            self.schedule_canvas.create_text(x, 45, text=time_pos.strftime("%H:%M"))

        # Draw current time indicator
        now_x = label_margin + ((now - start_time).total_seconds() / total_seconds) * timeline_width
        self.schedule_canvas.create_line(now_x, 40, now_x, canvas_height - 20, fill="red", dash=(2, 2))
        self.schedule_canvas.create_text(now_x, 15, text="NOW", fill="red")

        # Draw telescope lanes
        lane_height = (canvas_height - 80) / len(self.telescopes)
        for i, telescope in enumerate(self.telescopes):
            y = 70 + (i * lane_height)

            # Draw telescope name label
            self.schedule_canvas.create_text(
                label_margin - 10, y + lane_height / 2, text=telescope["name"], anchor="e"
            )

            # Draw current observation if any
            if telescope["current_observation"]:
                obs = telescope["current_observation"]
                start_x = label_margin + (
                    (obs.get("start_time_actual", obs["start_time"]) - start_time).total_seconds()
                    / total_seconds
                ) * timeline_width
                end_x = label_margin + (
                    (obs["end_time"] - start_time).total_seconds() / total_seconds
                ) * timeline_width
                end_x = max(end_x, start_x + 5)

                self.schedule_canvas.create_rectangle(
                    start_x, y + 5, end_x, y + lane_height - 5, fill="blue", outline="black"
                )
                self.schedule_canvas.create_text(
                    (start_x + end_x) / 2, y + lane_height / 2,
                    text=f"{obs['target']} ({obs['duration']}min)", fill="white"
                )

        # Draw scheduled observations
        for obs in self.schedule:
            if obs["status"] == "Scheduled" and obs["telescope"]:
                telescope_idx = next(i for i, t in enumerate(self.telescopes) if t["name"] == obs["telescope"])
                y = 70 + (telescope_idx * lane_height)

                start_x = label_margin + (
                    (obs["start_time"] - start_time).total_seconds() / total_seconds
                ) * timeline_width
                end_x = label_margin + (
                    (obs["end_time"] - start_time).total_seconds() / total_seconds
                ) * timeline_width
                end_x = max(end_x, start_x + 5)

                self.schedule_canvas.create_rectangle(
                    start_x, y + 5, end_x, y + lane_height - 5, fill="green", outline="black"
                )
                self.schedule_canvas.create_text(
                    (start_x + end_x) / 2, y + lane_height / 2,
                    text=f"{obs['target']} ({obs['duration']}min)", fill="white"
                )


    def update_history_display(self):
        """Update the history display"""
        self.history_text.delete(1.0, tk.END)

        if not self.history:
            self.history_text.insert(tk.END, "No observation history yet\n")
            return

        self.history_text.insert(tk.END, "=== Recent Observation History ===\n\n")

        for entry in sorted(
            self.history, key=lambda x: x["completed_at"], reverse=True
        )[:10]:
            status = "SUCCESS" if entry["success"] else "FAILED"
            color = "green" if entry["success"] else "red"

            self.history_text.insert(tk.END, f"Telescope: {entry['telescope']}\n")
            self.history_text.insert(
                tk.END, f"Observation: {entry['observation']['target']}\n"
            )
            self.history_text.insert(tk.END, f"Status: ")
            self.history_text.insert(tk.END, f"{status}\n", color)
            self.history_text.insert(
                tk.END,
                f"Completed: {entry['completed_at'].strftime('%Y-%m-%d %H:%M')}\n",
            )
            self.history_text.insert(
                tk.END, f"Duration: {entry['observation']['duration']} minutes\n"
            )
            self.history_text.insert(tk.END, "-" * 50 + "\n\n")

        # Configure text colors
        self.history_text.tag_config("green", foreground="green")
        self.history_text.tag_config("red", foreground="red")

    def update_stats_display(self):
        """Update the statistics display"""
        self.stats_ax.clear()

        if not self.history:
            self.stats_ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            self.stats_canvas.draw()
            return

        # Calculate some statistics
        success_rate = sum(1 for h in self.history if h["success"]) / len(self.history)
        telescope_usage = {
            t["name"]: sum(1 for h in self.history if h["telescope"] == t["name"])
            for t in self.telescopes
        }
        priority_dist = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}
        for h in self.history:
            priority = h["observation"]["priority"]
            priority_dist[priority] += 1

        # Create a bar chart of telescope usage
        telescopes = list(telescope_usage.keys())
        counts = list(telescope_usage.values())

        self.stats_ax.bar(telescopes, counts, color=["blue", "green", "orange"])
        self.stats_ax.set_title("Telescope Usage")
        self.stats_ax.set_ylabel("Number of Observations")

        # Add text info
        info_text = (
            f"Success Rate: {success_rate:.1%}\n"
            f"Critical: {priority_dist['Critical']} | High: {priority_dist['High']}\n"
            f"Medium: {priority_dist['Medium']} | Low: {priority_dist['Low']}"
        )

        self.stats_ax.text(
            0.5,
            -0.2,
            info_text,
            ha="center",
            va="center",
            transform=self.stats_ax.transAxes,
        )

        self.stats_canvas.draw()

    def update_comparison_display(self):
        """Update the telescope comparison display"""
        self.comp_ax1.clear()
        self.comp_ax2.clear()

        if not self.history:
            self.comp_ax1.text(0.5, 0.5, "No data available", ha="center", va="center")
            self.comp_canvas.draw()
            return

        # Prepare data for comparison
        telescopes = [t["name"] for t in self.telescopes]
        success_rates = []
        observation_counts = []
        observation_times = []

        for telescope in self.telescopes:
            total = telescope["success_count"] + telescope["failure_count"]
            if total > 0:
                success_rates.append(telescope["success_count"] / total)
            else:
                success_rates.append(0)

            observation_counts.append(
                telescope["success_count"] + telescope["failure_count"]
            )
            observation_times.append(telescope["total_observation_time"])

        # Plot success rates
        bars = self.comp_ax1.bar(
            telescopes, success_rates, color=["blue", "green", "orange"]
        )
        self.comp_ax1.set_title("Telescope Success Rates")
        self.comp_ax1.set_ylabel("Success Rate")
        self.comp_ax1.set_ylim(0, 1)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            self.comp_ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.1%}",
                ha="center",
                va="bottom",
            )

        # Plot observation counts and times
        width = 0.35
        x = range(len(telescopes))

        bars1 = self.comp_ax2.bar(
            x, observation_counts, width, label="Observation Count"
        )
        bars2 = self.comp_ax2.bar(
            [i + width for i in x],
            observation_times,
            width,
            label="Total Observation Time (min)",
        )

        self.comp_ax2.set_title("Telescope Utilization")
        self.comp_ax2.set_ylabel("Count/Time")
        self.comp_ax2.set_xticks([i + width / 2 for i in x])
        self.comp_ax2.set_xticklabels(telescopes)
        self.comp_ax2.legend()

        # Add value labels on bars
        for bar in bars1 + bars2:
            height = bar.get_height()
            self.comp_ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
            )

        # Determine which telescope is best
        best_telescope = None
        best_score = -1

        for i, telescope in enumerate(self.telescopes):
            # Simple scoring: success rate * observation count
            score = success_rates[i] * observation_counts[i]
            if score > best_score:
                best_score = score
                best_telescope = telescope["name"]

        # Add best telescope annotation
        # self.comp_ax1.text(0.5, -0.2,
        #                   f"Best Performing Telescope: {best_telescope} (Score: {best_score:.1f})",
        #                   ha='center', va='center', transform=self.comp_ax1.transAxes,
        #                   fontsize=10, bbox=dict(facecolor='yellow', alpha=0.5))

        # self.comp_canvas.draw()

    def _on_log_scroll(self, *args):
        # Pass scroll values to the scrollbar
        self.log_console_scrollbar.set(*args)

        # Determine if the scroll is at the bottom
        try:
            end = float(args[1]) if args[0] == 'moveto' else self.log_console.yview()[1]
            self._log_at_bottom = end >= 0.999
        except Exception:
            self._log_at_bottom = True  # Safe default

    def log(self, message, tag=None):
        self.log_console.configure(state=tk.NORMAL)
        if tag:
            self.log_console.insert(tk.END, message + "\n", tag)
        else:
            self.log_console.insert(tk.END, message + "\n")
        if self._log_at_bottom:
            self.log_console.see(tk.END)
        self.log_console.configure(state=tk.DISABLED)



if __name__ == "__main__":
    root = tk.Tk()
    try:
        app = RealTimeTelescopeScheduler(root)
        root.mainloop()
    except Exception as e:
        messagebox.showerror("Fatal Error", f"Application crashed: {str(e)}")
        raise
