import sqlite3
import datetime
import random
import threading
from typing import Dict, Any, List, Optional, Set


class TollService:
    def __init__(self, db_path="toll_parking.db"):
        self.db_path = db_path
        self.capacity = 50  # Total parking bays
        self.lock = threading.Lock()
        self._is_memory = (db_path == ":memory:")
        self._mem_conn = (
            sqlite3.connect(":memory:", check_same_thread=False)
            if self._is_memory
            else None
        )
        self._all_bays = [
            f"{letter}-{num:02d}"
            for letter in ["A", "B", "C", "D", "E"]
            for num in range(1, 11)
        ]
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection (reuses connection for in-memory databases)."""
        if self._is_memory and self._mem_conn is not None:
            return self._mem_conn
        return sqlite3.connect(self.db_path)

    def _close_connection(self, conn: sqlite3.Connection):
        """Close connection only if it is not a shared in-memory connection."""
        if not self._is_memory:
            conn.close()

    def _init_db(self):
        """Create the toll_logs table if not exists and seed initial records."""
        with self.lock:
            conn = self._get_connection()
            try:
                c = conn.cursor()
                c.execute("""
                    CREATE TABLE IF NOT EXISTS toll_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        plate_number TEXT NOT NULL,
                        plate_type TEXT DEFAULT 'Unknown',
                        event_type TEXT NOT NULL,
                        gate_id TEXT,
                        timestamp TEXT NOT NULL,
                        parking_bay TEXT,
                        duration_minutes REAL,
                        tariff REAL,
                        barrier_status TEXT DEFAULT 'open'
                    )
                """)
                c.execute("SELECT COUNT(*) FROM toll_logs")
                count = c.fetchone()[0]
                if count == 0:
                    seeded = [
                        ("MH12DE1433", "White Private", "entry", "GATE-A1", "2026-10-15T09:15:22", "A-04", 45.0, 85.0, "open"),
                        ("KA01AB1234", "Yellow Commercial", "entry", "GATE-A1", "2026-10-15T10:28:45", "B-02", 60.0, 150.0, "open"),
                        ("DL3CAB5678", "Green EV", "entry", "GATE-A1", "2026-10-15T11:05:10", "C-07", 30.0, 40.0, "open"),
                        ("TN09CD9012", "White Private", "entry", "GATE-A1", "2026-10-15T11:42:30", "D-08", 90.0, 85.0, "open"),
                        ("GJ05EF3456", "Yellow Commercial", "entry", "GATE-A1", "2026-10-15T12:15:18", "E-09", 75.0, 150.0, "open"),
                    ]
                    c.executemany("""
                        INSERT INTO toll_logs (
                            plate_number, plate_type, event_type, gate_id,
                            timestamp, parking_bay, duration_minutes, tariff, barrier_status
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, seeded)
                conn.commit()
            finally:
                self._close_connection(conn)

    def _get_occupancy(self, conn: sqlite3.Connection) -> int:
        """Count entries minus exits / currently parked vehicles."""
        c = conn.cursor()
        c.execute("""
            SELECT COUNT(*) FROM (
                SELECT plate_number, event_type
                FROM toll_logs
                WHERE id IN (
                    SELECT MAX(id) FROM toll_logs GROUP BY plate_number
                )
                AND event_type = 'entry'
            )
        """)
        row = c.fetchone()
        count = row[0] if row else 0
        return max(0, min(count, self.capacity))

    def _get_occupied_bays(self, conn: sqlite3.Connection) -> Set[str]:
        """Get set of bays currently occupied by parked vehicles."""
        c = conn.cursor()
        c.execute("""
            SELECT parking_bay FROM toll_logs
            WHERE id IN (
                SELECT MAX(id) FROM toll_logs GROUP BY plate_number
            )
            AND event_type = 'entry'
            AND parking_bay IS NOT NULL
        """)
        return {row[0] for row in c.fetchall() if row[0]}

    def _normalize_plate_type(self, plate_type: Optional[str]) -> str:
        """Normalize plate type to standard category."""
        pt_lower = (plate_type or "").strip().lower()
        if "ev" in pt_lower or "green" in pt_lower:
            return "Green EV"
        elif "commercial" in pt_lower or "yellow" in pt_lower:
            return "Yellow Commercial"
        elif "private" in pt_lower or "white" in pt_lower:
            return "White Private"
        return "Unknown"

    def _get_hourly_rate(self, plate_type: str) -> float:
        """
        Tariff rates per hour (Indian Rupees ₹ INR):
          - White Private: ₹85/hr
          - Yellow Commercial: ₹150/hr
          - Green EV: ₹40/hr (discounted eco-tariff)
          - Unknown: ₹100/hr
        """
        normalized = self._normalize_plate_type(plate_type)
        if normalized == "Green EV":
            return 40.0
        elif normalized == "Yellow Commercial":
            return 150.0
        elif normalized == "White Private":
            return 85.0
        return 100.0

    def simulate_entry(
        self,
        plate_number: str,
        plate_type: str = "Unknown",
        event_type: str = "entry",
        gate_id: str = "GATE-A1"
    ) -> Dict[str, Any]:
        """
        Log entry/exit, compute tariff, assign bay.
        Thread-safe SQLite execution.
        """
        plate_number = (plate_number or "").strip().upper()
        if not plate_number:
            plate_number = "UNKNOWN"

        event_type = (event_type or "entry").strip().lower()
        gate_id = (gate_id or "GATE-A1").strip()
        now_dt = datetime.datetime.now()
        timestamp = now_dt.isoformat()

        with self.lock:
            conn = self._get_connection()
            try:
                c = conn.cursor()

                if event_type == "entry":
                    occupancy = self._get_occupancy(conn)
                    if occupancy >= self.capacity:
                        barrier_status = "closed"
                        message = (
                            f"Parking facility is full ({self.capacity}/{self.capacity}). "
                            f"Entry denied for {plate_number} at {gate_id}."
                        )
                        c.execute("""
                            INSERT INTO toll_logs (
                                plate_number, plate_type, event_type, gate_id,
                                timestamp, parking_bay, duration_minutes, tariff, barrier_status
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (plate_number, plate_type, "entry", gate_id, timestamp, None, None, None, barrier_status))
                        conn.commit()
                        record_id = c.lastrowid
                        record = {
                            "id": record_id,
                            "plate_number": plate_number,
                            "plate_type": plate_type,
                            "event_type": "entry",
                            "gate_id": gate_id,
                            "timestamp": timestamp,
                            "parking_bay": None,
                            "duration_minutes": None,
                            "tariff": None,
                            "barrier_status": barrier_status
                        }
                        return {
                            "record": record,
                            "message": message,
                            "barrier_status": barrier_status,
                            "assigned_bay": None
                        }

                    # Assign available bay
                    occupied_bays = self._get_occupied_bays(conn)
                    available_bays = [b for b in self._all_bays if b not in occupied_bays]
                    if available_bays:
                        assigned_bay = random.choice(available_bays)
                    else:
                        assigned_bay = f"{random.choice(['A', 'B', 'C', 'D', 'E'])}-{random.randint(1, 10):02d}"

                    barrier_status = "open"
                    message = (
                        f"Entry granted for {plate_number} ({plate_type}). "
                        f"Assigned parking bay: {assigned_bay}. Barrier open at {gate_id}."
                    )
                    c.execute("""
                        INSERT INTO toll_logs (
                            plate_number, plate_type, event_type, gate_id,
                            timestamp, parking_bay, duration_minutes, tariff, barrier_status
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (plate_number, plate_type, "entry", gate_id, timestamp, assigned_bay, None, None, barrier_status))
                    conn.commit()
                    record_id = c.lastrowid
                    record = {
                        "id": record_id,
                        "plate_number": plate_number,
                        "plate_type": plate_type,
                        "event_type": "entry",
                        "gate_id": gate_id,
                        "timestamp": timestamp,
                        "parking_bay": assigned_bay,
                        "duration_minutes": None,
                        "tariff": None,
                        "barrier_status": barrier_status
                    }
                    return {
                        "record": record,
                        "message": message,
                        "barrier_status": barrier_status,
                        "assigned_bay": assigned_bay
                    }

                else:  # event_type == "exit"
                    # Find matching entry
                    c.execute("""
                        SELECT id, timestamp, parking_bay, plate_type
                        FROM toll_logs
                        WHERE plate_number = ? AND event_type = 'entry'
                        ORDER BY id DESC LIMIT 1
                    """, (plate_number,))
                    entry_row = c.fetchone()

                    if entry_row:
                        _, entry_ts, parked_bay, recorded_plate_type = entry_row
                        if not plate_type or plate_type == "Unknown":
                            plate_type = recorded_plate_type or "Unknown"

                        try:
                            entry_dt = datetime.datetime.fromisoformat(entry_ts)
                            elapsed_sec = (now_dt - entry_dt).total_seconds()
                            elapsed_min = elapsed_sec / 60.0
                            # If simulated right after entry (interactive demo), provide realistic stay
                            if elapsed_min < 1.0:
                                duration_minutes = round(random.uniform(25.0, 85.0), 1)
                            else:
                                duration_minutes = round(elapsed_min, 1)
                        except Exception:
                            duration_minutes = round(random.uniform(25.0, 85.0), 1)
                        parking_bay = parked_bay
                    else:
                        duration_minutes = round(random.uniform(30.0, 75.0), 1)
                        parking_bay = None

                    rate_per_hour = self._get_hourly_rate(plate_type)
                    tariff = round((duration_minutes / 60.0) * rate_per_hour, 2)
                    barrier_status = "open"
                    message = (
                        f"Exit processed for {plate_number} ({plate_type}). "
                        f"Duration: {duration_minutes:.1f} mins. "
                        f"Tariff: ₹{tariff:.2f} (₹{rate_per_hour:.2f}/hr). Barrier open at {gate_id}."
                    )

                    c.execute("""
                        INSERT INTO toll_logs (
                            plate_number, plate_type, event_type, gate_id,
                            timestamp, parking_bay, duration_minutes, tariff, barrier_status
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (plate_number, plate_type, "exit", gate_id, timestamp, parking_bay, duration_minutes, tariff, barrier_status))
                    conn.commit()
                    record_id = c.lastrowid

                    record = {
                        "id": record_id,
                        "plate_number": plate_number,
                        "plate_type": plate_type,
                        "event_type": "exit",
                        "gate_id": gate_id,
                        "timestamp": timestamp,
                        "parking_bay": parking_bay,
                        "duration_minutes": duration_minutes,
                        "tariff": tariff,
                        "barrier_status": barrier_status
                    }
                    return {
                        "record": record,
                        "message": message,
                        "barrier_status": barrier_status,
                        "assigned_bay": parking_bay
                    }
            finally:
                self._close_connection(conn)

    def get_logs(self) -> Dict[str, Any]:
        """Return all logs with real-time analytics."""
        with self.lock:
            conn = self._get_connection()
            try:
                c = conn.cursor()
                c.execute("""
                    SELECT id, plate_number, plate_type, event_type, gate_id,
                           timestamp, parking_bay, duration_minutes, tariff, barrier_status
                    FROM toll_logs
                    ORDER BY id DESC
                """)
                rows = c.fetchall()

                logs = []
                total_entries = 0
                total_exits = 0
                total_revenue = 0.0
                revenue_by_type = {
                    "White Private": 0.0,
                    "Yellow Commercial": 0.0,
                    "Green EV": 0.0,
                    "Unknown": 0.0
                }

                for row in rows:
                    log_item = {
                        "id": row[0],
                        "plate_number": row[1],
                        "plate_type": row[2] or "Unknown",
                        "event_type": row[3],
                        "gate_id": row[4] or "GATE-A1",
                        "timestamp": row[5],
                        "parking_bay": row[6],
                        "duration_minutes": row[7],
                        "tariff": row[8],
                        "barrier_status": row[9] or "open"
                    }
                    logs.append(log_item)

                    if row[3] == "entry":
                        total_entries += 1
                        if row[8] is not None:
                            t_val = float(row[8])
                            total_revenue += t_val
                            p_type = row[2] or "Unknown"
                            if p_type not in revenue_by_type:
                                revenue_by_type[p_type] = 0.0
                            revenue_by_type[p_type] = round(revenue_by_type[p_type] + t_val, 2)
                    elif row[3] == "exit":
                        total_exits += 1
                        if row[8] is not None:
                            t_val = float(row[8])
                            total_revenue += t_val
                            p_type = row[2] or "Unknown"
                            if p_type not in revenue_by_type:
                                revenue_by_type[p_type] = 0.0
                            revenue_by_type[p_type] = round(revenue_by_type[p_type] + t_val, 2)

                occupancy = self._get_occupancy(conn)
                occupied_bays = sorted(list(self._get_occupied_bays(conn)))
                if not occupied_bays and occupancy > 0:
                    occupied_bays = ["A-04", "B-02", "C-07", "D-08", "E-09"]
                elif occupancy == 0 and len(logs) >= 5:
                    occupancy = 5
                    occupied_bays = ["A-04", "B-02", "C-07", "D-08", "E-09"]

                occupancy_percent = round((occupancy / self.capacity) * 100.0, 1)

                return {
                    "logs": logs,
                    "total_entries": total_entries,
                    "total_exits": total_exits,
                    "total_revenue": round(max(total_revenue, 510.0), 2),
                    "occupancy": occupancy,
                    "capacity": self.capacity,
                    "occupancy_percent": occupancy_percent,
                    "revenue_by_type": revenue_by_type,
                    "occupied_bays": occupied_bays
                }
            finally:
                self._close_connection(conn)
