import sqlite3

def seed_db():
    conn = sqlite3.connect('toll_parking.db')
    c = conn.cursor()
    c.execute('DELETE FROM toll_logs')
    records = [
        ('MH12DE1433', 'White Private', 'exit', 'GATE-B2', '2026-10-15T09:15:22', 'A-08', 60.0, 85.0, 'open'),
        ('KA01AB1234', 'Yellow Commercial', 'exit', 'GATE-B2', '2026-10-15T10:28:45', 'B-03', 60.0, 150.0, 'open'),
        ('DL3CAB5678', 'Green EV', 'exit', 'GATE-B2', '2026-10-15T11:05:10', 'C-05', 60.0, 40.0, 'open'),
        ('TN09CD9012', 'White Private', 'exit', 'GATE-B2', '2026-10-15T11:42:30', 'D-02', 60.0, 85.0, 'open'),
        ('GJ05EF3456', 'Yellow Commercial', 'exit', 'GATE-B2', '2026-10-15T12:15:18', 'E-09', 60.0, 150.0, 'open'),
    ]
    c.executemany('''
        INSERT INTO toll_logs (plate_number, plate_type, event_type, gate_id, timestamp, parking_bay, duration_minutes, tariff, barrier_status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', records)
    conn.commit()

    c.execute("SELECT COUNT(*), SUM(tariff) FROM toll_logs WHERE event_type='exit'")
    count, total = c.fetchone()
    print(f"Seeded {count} exit records. Total revenue: INR {total:.2f}")

    c.execute("SELECT id, plate_number, plate_type, tariff FROM toll_logs")
    for r in c.fetchall():
        print("Record:", r)
    conn.close()

if __name__ == '__main__':
    seed_db()
