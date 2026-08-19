from dotenv import load_dotenv
load_dotenv()
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
import math
import uuid
import sqlite3
from datetime import datetime

import joblib
import pandas as pd
from flask import (Flask, render_template, request, jsonify,
                   redirect, session, url_for)
from werkzeug.security import generate_password_hash, check_password_hash

# ── CONFIG ───────────────────────────────────────────────────────────────────

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DB_PATH    = os.path.join(BASE_DIR, "routes.db")
MODEL_PATH = os.path.join(BASE_DIR, "ml", "models", "anomaly_model.pkl")
SCALER_PATH= os.path.join(BASE_DIR, "ml", "models", "scaler.pkl")

TRIP_GAP_MINUTES      = 15
ZONE_PRECISION        = 3
PATTERN_TRIP_THRESHOLD= 3
KNOWN_AREA_RADIUS_DEG = 0.003
MIN_MOVEMENT_KM = 0.3        # must move ~300m from trip start before flagging a route deviation
ONBOARDING_GRACE_PINGS = 5   # first few pings on a brand-new account are auto-SAFE while it learns
MIN_TRIP_DISTANCE_KM  = 0.05

# Twilio — set these in environment variables, never hardcode
TWILIO_SID   = os.environ.get("TWILIO_ACCOUNT_SID", "")
TWILIO_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM  = os.environ.get("TWILIO_PHONE_NUMBER", "")   # e.g. +15551234567
TWILIO_ENABLED = bool(TWILIO_SID and TWILIO_TOKEN and TWILIO_FROM)

GMAIL_ADDRESS = os.environ.get("PATHPULSE_GMAIL", "")
GMAIL_APP_PASS = os.environ.get("PATHPULSE_GMAIL_APP_PASSWORD", "")
EMAIL_ENABLED = bool(GMAIL_ADDRESS and GMAIL_APP_PASS)

app = Flask(__name__)
app.secret_key = os.environ.get("PATHPULSE_SECRET_KEY", os.urandom(24).hex())

# ── ML MODEL ─────────────────────────────────────────────────────────────────

try:
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
    ML_READY = True
    print("[ML] model loaded:", type(model).__name__)
except Exception as e:
    model = scaler = None
    ML_READY = False
    print("[ML] no model found, rule-based only:", e)

FEATURES = ["latitude","longitude","hour","dayofweek","speed_kmh",
            "is_night","sudden_stop","nearest_risk_km","area_risk_score"]

def compute_ml_score(lat, lon, hour, dow, speed_kmh,
                     is_night=0, sudden_stop=0,
                     nearest_risk_km=999.0, area_risk_score=0.0):
    if not ML_READY:
        return False, 0.0
    X = pd.DataFrame([[lat, lon, hour, dow, speed_kmh,
                        is_night, sudden_stop, nearest_risk_km, area_risk_score]],
                     columns=FEATURES)
    if scaler is not None:
        X = scaler.transform(X)
    try:
        pred  = model.predict(X)[0]
        score = float(model.decision_function(X)[0])
        return pred == -1, score
    except Exception as e:
        print("[ML] scoring failed:", e)
        return False, 0.0

# ── DB HELPERS ────────────────────────────────────────────────────────────────

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn

def init_db():
    conn = get_db()
    conn.execute("""
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        code_word_hash TEXT,
        share_token TEXT UNIQUE,
        created_at TEXT
    )""")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS contacts(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER, name TEXT, phone TEXT, relationship TEXT
    )""")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS routes(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER, trip_id INTEGER,
        latitude REAL, longitude REAL, speed_kmh REAL, time TEXT
    )""")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS trips(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        start_lat REAL, start_lon REAL, end_lat REAL, end_lon REAL,
        start_time TEXT, end_time TEXT,
        distance_km REAL DEFAULT 0, duration_min REAL DEFAULT 0,
        status TEXT DEFAULT 'active', confirmed_safe INTEGER DEFAULT 0
    )""")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS daily_patterns(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER, start_zone TEXT, end_zone TEXT,
        trip_count INTEGER DEFAULT 1, status TEXT DEFAULT 'LEARNING',
        last_traveled TEXT,
        UNIQUE(user_id, start_zone, end_zone)
    )""")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS alerts(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER, latitude REAL, longitude REAL,
        reason TEXT, time TEXT, resolved INTEGER DEFAULT 0
    )""")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS verified_routes(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER, latitude REAL, longitude REAL,
        safe_count INTEGER DEFAULT 1
    )""")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS anomalies(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER, latitude REAL, longitude REAL,
        anomaly_score REAL, reason TEXT, time TEXT
    )""")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS risk_zones(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        latitude REAL, longitude REAL, risk_score REAL, source TEXT
    )""")
    conn.commit()
    conn.close()

init_db()

# ── GEO ───────────────────────────────────────────────────────────────────────

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1,p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2-lat1); dl = math.radians(lon2-lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*R*math.asin(math.sqrt(a))

def zone_key(lat, lon):
    return f"{round(lat,ZONE_PRECISION)},{round(lon,ZONE_PRECISION)}"

# ── TRIP / PATTERN LOGIC ──────────────────────────────────────────────────────

def get_active_trip(conn, user_id):
    return conn.execute(
        "SELECT * FROM trips WHERE user_id=? AND status='active' ORDER BY id DESC LIMIT 1",
        (user_id,)).fetchone()

def start_trip(conn, user_id, lat, lon, now):
    cur = conn.execute(
        "INSERT INTO trips(user_id,start_lat,start_lon,end_lat,end_lon,start_time,end_time,status)"
        " VALUES(?,?,?,?,?,?,?,'active')",
        (user_id,lat,lon,lat,lon,now.isoformat(),now.isoformat()))
    return cur.lastrowid

def touch_trip(conn, trip_id, lat, lon, now):
    conn.execute("UPDATE trips SET end_lat=?,end_lon=?,end_time=? WHERE id=?",
                 (lat,lon,now.isoformat(),trip_id))

def increment_pattern(conn, user_id, start_zone, end_zone):
    row = conn.execute(
        "SELECT * FROM daily_patterns WHERE user_id=? AND start_zone=? AND end_zone=?",
        (user_id,start_zone,end_zone)).fetchone()
    now = datetime.now().isoformat()
    if row:
        new_count = row["trip_count"]+1
        status = "NORMAL" if new_count >= PATTERN_TRIP_THRESHOLD else "LEARNING"
        conn.execute("UPDATE daily_patterns SET trip_count=?,status=?,last_traveled=? WHERE id=?",
                     (new_count,status,now,row["id"]))
    else:
        conn.execute(
            "INSERT INTO daily_patterns(user_id,start_zone,end_zone,trip_count,status,last_traveled)"
            " VALUES(?,?,?,1,'LEARNING',?)",
            (user_id,start_zone,end_zone,now))

def finalize_trip(conn, trip):
    start_zone = zone_key(trip["start_lat"],trip["start_lon"])
    end_zone   = zone_key(trip["end_lat"],trip["end_lon"])
    distance   = haversine_km(trip["start_lat"],trip["start_lon"],
                               trip["end_lat"],trip["end_lon"])
    start_t = datetime.fromisoformat(trip["start_time"])
    end_t   = datetime.fromisoformat(trip["end_time"])
    duration_min = max((end_t-start_t).total_seconds()/60.0, 0.1)
    conn.execute("UPDATE trips SET status='completed',distance_km=?,duration_min=? WHERE id=?",
                 (distance,duration_min,trip["id"]))
    if distance >= MIN_TRIP_DISTANCE_KM:
        increment_pattern(conn,trip["user_id"],start_zone,end_zone)

def known_destinations_for_start(conn, user_id, start_zone):
    rows = conn.execute(
        "SELECT end_zone FROM daily_patterns WHERE user_id=? AND start_zone=? AND status='NORMAL'",
        (user_id,start_zone)).fetchall()
    return {r["end_zone"] for r in rows}

# ── TWILIO SMS ────────────────────────────────────────────────────────────────

def send_sms_alert(user_id, lat, lon, reason, conn):
    """Send SMS to all emergency contacts via Twilio."""
    if not TWILIO_ENABLED:
        return
    try:
        from twilio.rest import Client
        client = Client(TWILIO_SID, TWILIO_TOKEN)
        contacts = conn.execute(
            "SELECT name, phone FROM contacts WHERE user_id=?", (user_id,)
        ).fetchall()
        user = conn.execute("SELECT name FROM users WHERE id=?", (user_id,)).fetchone()
        user_name = user["name"] if user else "Someone"
        maps_link = f"https://maps.google.com/?q={lat},{lon}"
        msg = (f"🚨 PATH PULSE ALERT\n"
               f"{user_name} may need help.\n"
               f"Reason: {reason}\n"
               f"Location: {maps_link}\n"
               f"Please check on them immediately.")
        for contact in contacts:
            phone = contact["phone"].strip()
            if not phone.startswith("+"):
                phone = "+91" + phone.lstrip("0")
            try:
                client.messages.create(body=msg, from_=TWILIO_FROM, to=phone)
                print(f"[SMS] sent to {contact['name']} at {phone}")
            except Exception as e:
                print(f"[SMS] failed to send to {phone}: {e}")
    except ImportError:
        print("[SMS] twilio not installed — run: pip install twilio")
    except Exception as e:
        print(f"[SMS] error: {e}")

def send_email_alert(user_id, lat, lon, reason, conn, photo_bytes=None):
    if not EMAIL_ENABLED:
        print("[EMAIL] Not configured.")
        return
    try:
        contacts = conn.execute(
            "SELECT name, phone FROM contacts WHERE user_id=?", (user_id,)
        ).fetchall()
        user = conn.execute("SELECT name FROM users WHERE id=?", (user_id,)).fetchone()
        user_name = user["name"] if user else "Someone"
        maps_link = f"https://maps.google.com/?q={lat},{lon}"
        subject = f"PATH PULSE ALERT - {user_name} may need help"
        photo_html = ""
        if photo_bytes:
            photo_html = ("<div style='margin:16px 0;'>"
                          "<p style='margin:0 0 8px;color:#9aa0b8;font-size:13px;'>SURROUNDING PHOTO</p>"
                          "<img src='cid:sos_photo' style='width:100%;border-radius:8px;display:block;'>"
                          "</div>")
        body = ("<html><body style='font-family:Arial,sans-serif;max-width:600px;margin:0 auto;'>"
                "<div style='background:#ef4444;padding:20px;border-radius:12px 12px 0 0;'>"
                "<h1 style='color:white;margin:0;'>Path Pulse SOS Alert</h1></div>"
                "<div style='background:#1a1a2e;padding:24px;border-radius:0 0 12px 12px;color:#e0e0e0;'>"
                f"<p style='font-size:18px;'><strong style='color:white;'>{user_name}</strong> may need immediate help.</p>"
                "<div style='background:#2a2a4a;padding:16px;border-radius:8px;margin:16px 0;'>"
                "<p style='margin:0;color:#9aa0b8;font-size:13px;'>REASON</p>"
                f"<p style='margin:6px 0 0;color:white;'>{reason}</p></div>"
                f"{photo_html}"
                f"<a href='{maps_link}' style='display:block;background:#2dd4bf;color:#04201c;padding:14px;"
                "border-radius:8px;text-align:center;font-weight:bold;text-decoration:none;margin:20px 0;'>"
                "Open Location in Google Maps</a>"
                "<p style='color:#9aa0b8;font-size:13px;text-align:center;'>"
                f"Sent automatically by Path Pulse. Please check on {user_name} immediately.</p>"
                "</div></body></html>")
        recipients = []
        for c in contacts:
            if "@" in c["phone"]:
                recipients.append(c["phone"].strip())
        if not recipients:
            recipients = [GMAIL_ADDRESS]
        recipients = list(set(recipients))
        msg = MIMEMultipart("related")
        msg["Subject"] = subject
        msg["From"] = "Path Pulse <" + GMAIL_ADDRESS + ">"
        msg["To"] = ", ".join(recipients)
        alt = MIMEMultipart("alternative")
        alt.attach(MIMEText(body, "html"))
        msg.attach(alt)
        if photo_bytes:
            from email.mime.image import MIMEImage
            img = MIMEImage(photo_bytes)
            img.add_header("Content-ID", "<sos_photo>")
            img.add_header("Content-Disposition", "inline", filename="surroundings.jpg")
            msg.attach(img)
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(GMAIL_ADDRESS, GMAIL_APP_PASS)
            smtp.sendmail(GMAIL_ADDRESS, recipients, msg.as_string())
        print("[EMAIL] Alert sent to:", recipients, "with photo" if photo_bytes else "no photo")
    except Exception as e:
        print("[EMAIL] Failed:", e)


def insert_alert(conn, user_id, lat, lon, reason, send_sms=True, photo_bytes=None):
    conn.execute(
        "INSERT INTO alerts(user_id,latitude,longitude,reason,time,resolved)"
        " VALUES(?,?,?,?,?,0)",
        (user_id,lat,lon,reason,datetime.now().isoformat()))
    conn.commit()
    if send_sms:
        send_sms_alert(user_id,lat,lon,reason,conn)
    send_email_alert(user_id,lat,lon,reason,conn,photo_bytes=photo_bytes)

# ── PAGES ─────────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect("/login")
    return render_template("dashboard.html", name=session.get("name"))

@app.route("/history")
def history():
    if "user_id" not in session:
        return redirect("/login")
    conn = get_db()
    rows = conn.execute(
        "SELECT latitude,longitude,speed_kmh,time FROM routes"
        " WHERE user_id=? ORDER BY id DESC LIMIT 500",
        (session["user_id"],)).fetchall()
    trips = conn.execute(
        "SELECT * FROM trips WHERE user_id=? AND status='completed' ORDER BY id DESC LIMIT 50",
        (session["user_id"],)).fetchall()
    conn.close()
    return render_template("history.html", rows=rows, trips=trips)

@app.route("/analytics")
def analytics():
    if "user_id" not in session:
        return redirect("/login")
    uid = session["user_id"]
    conn = get_db()
    total_trips    = conn.execute("SELECT COUNT(*) c FROM trips WHERE user_id=?", (uid,)).fetchone()["c"]
    new_routes     = conn.execute("SELECT COUNT(*) c FROM anomalies WHERE user_id=?", (uid,)).fetchone()["c"]
    sos_alerts     = conn.execute("SELECT COUNT(*) c FROM alerts WHERE user_id=?", (uid,)).fetchone()["c"]
    safe_routes    = max(total_trips - new_routes, 0)
    normal_patterns= conn.execute(
        "SELECT COUNT(*) c FROM daily_patterns WHERE user_id=? AND status='NORMAL'", (uid,)
    ).fetchone()["c"]
    daily = list(reversed(conn.execute(
        "SELECT date(start_time) d, COUNT(*) c FROM trips"
        " WHERE user_id=? GROUP BY d ORDER BY d DESC LIMIT 7", (uid,)
    ).fetchall()))
    recent_alerts = conn.execute(
        "SELECT * FROM alerts WHERE user_id=? ORDER BY id DESC LIMIT 10", (uid,)
    ).fetchall()
    conn.close()
    return render_template("analytics.html",
        total_trips=total_trips, safe_routes=safe_routes,
        new_routes=new_routes, sos_alerts=sos_alerts,
        normal_patterns=normal_patterns,
        daily_labels=[r["d"] for r in daily],
        daily_counts=[r["c"] for r in daily],
        recent_alerts=recent_alerts, ml_ready=ML_READY,
        twilio_enabled=TWILIO_ENABLED)

# ── SETTINGS (Feature 3 & 4) ─────────────────────────────────────────────────

@app.route("/settings", methods=["GET","POST"])
def settings():
    if "user_id" not in session:
        return redirect("/login")
    uid = session["user_id"]
    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
    message = None
    error   = None

    if request.method == "POST":
        action = request.form.get("action")

        if action == "update_profile":
            name = request.form.get("name","").strip()
            if not name:
                error = "Name cannot be empty."
            else:
                conn.execute("UPDATE users SET name=? WHERE id=?", (name,uid))
                conn.commit()
                session["name"] = name
                message = "Name updated successfully."

        elif action == "update_password":
            current  = request.form.get("current_password","")
            new_pass = request.form.get("new_password","")
            confirm  = request.form.get("confirm_password","")
            if not check_password_hash(user["password_hash"], current):
                error = "Current password is incorrect."
            elif len(new_pass) < 6:
                error = "New password must be at least 6 characters."
            elif new_pass != confirm:
                error = "New passwords do not match."
            else:
                conn.execute("UPDATE users SET password_hash=? WHERE id=?",
                             (generate_password_hash(new_pass), uid))
                conn.commit()
                message = "Password updated successfully."

        elif action == "update_code_word":
            new_word = request.form.get("new_code_word","").strip().lower()
            if not new_word:
                error = "Code word cannot be empty."
            else:
                conn.execute("UPDATE users SET code_word_hash=? WHERE id=?",
                             (generate_password_hash(new_word), uid))
                conn.commit()
                message = "Secret code word updated."

        # reload user after changes
        user = conn.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()

    conn.close()
    return render_template("settings.html", user=user,
                           message=message, error=error,
                           twilio_enabled=TWILIO_ENABLED)

# ── REAL-TIME SHARE LINK (Feature 2) ─────────────────────────────────────────

@app.route("/share/generate", methods=["POST"])
def generate_share_link():
    if "user_id" not in session:
        return jsonify({"error": "Login required"}), 401
    token = uuid.uuid4().hex
    conn = get_db()
    conn.execute("UPDATE users SET share_token=? WHERE id=?",
                 (token, session["user_id"]))
    conn.commit()
    conn.close()
    share_url = request.host_url + "share/" + token
    return jsonify({"url": share_url})

@app.route("/share/revoke", methods=["POST"])
def revoke_share_link():
    if "user_id" not in session:
        return jsonify({"error": "Login required"}), 401
    conn = get_db()
    conn.execute("UPDATE users SET share_token=NULL WHERE id=?",
                 (session["user_id"],))
    conn.commit()
    conn.close()
    return jsonify({"message": "Link revoked"})

@app.route("/share/<token>")
def live_share(token):
    """Public page — no login needed — shows the user's live location."""
    conn = get_db()
    user = conn.execute("SELECT id,name FROM users WHERE share_token=?",
                        (token,)).fetchone()
    if not user:
        conn.close()
        return render_template("share_expired.html"), 404
    last = conn.execute(
        "SELECT latitude,longitude,time FROM routes"
        " WHERE user_id=? ORDER BY id DESC LIMIT 1",
        (user["id"],)).fetchone()
    conn.close()
    return render_template("live_share.html", user=user,
                           last=last, token=token)

@app.route("/share/<token>/data")
def live_share_data(token):
    """Polled by the live_share page every 10s to get the latest location."""
    conn = get_db()
    user = conn.execute("SELECT id FROM users WHERE share_token=?",
                        (token,)).fetchone()
    if not user:
        conn.close()
        return jsonify({"error": "expired"}), 404
    last = conn.execute(
        "SELECT latitude,longitude,time FROM routes"
        " WHERE user_id=? ORDER BY id DESC LIMIT 1",
        (user["id"],)).fetchone()
    conn.close()
    if not last:
        return jsonify({"status": "no_data"})
    return jsonify({"lat": last["latitude"],
                    "lon": last["longitude"],
                    "time": last["time"]})

# ── CONTACTS ──────────────────────────────────────────────────────────────────

@app.route("/contacts", methods=["GET","POST"])
def contacts():
    if "user_id" not in session:
        return redirect("/login")
    conn = get_db()
    if request.method == "POST":
        conn.execute(
            "INSERT INTO contacts(user_id,name,phone,relationship) VALUES(?,?,?,?)",
            (session["user_id"],
             request.form["name"].strip(),
             request.form["phone"].strip(),
             request.form["relationship"].strip()))
        conn.commit()
    rows = conn.execute(
        "SELECT id,name,phone,relationship FROM contacts WHERE user_id=?",
        (session["user_id"],)).fetchall()
    conn.close()
    return render_template("contacts.html", rows=rows)

@app.route("/delete_contact/<int:id>")
def delete_contact(id):
    if "user_id" not in session:
        return redirect("/login")
    conn = get_db()
    conn.execute("DELETE FROM contacts WHERE id=? AND user_id=?",
                 (id, session["user_id"]))
    conn.commit()
    conn.close()
    return redirect("/contacts")

# ── AUTH ──────────────────────────────────────────────────────────────────────

@app.route("/register", methods=["GET","POST"])
def register():
    if request.method == "POST":
        name      = request.form["name"].strip()
        email     = request.form["email"].strip().lower()
        password  = request.form["password"]
        code_word = request.form.get("code_word","").strip().lower()
        if not name or not email:
            return render_template("register.html", error="Name and email are required.")
        if len(password) < 6:
            return render_template("register.html", error="Password must be at least 6 characters.")
        conn = get_db()
        try:
            conn.execute(
                "INSERT INTO users(name,email,password_hash,code_word_hash,created_at)"
                " VALUES(?,?,?,?,?)",
                (name, email,
                 generate_password_hash(password),
                 generate_password_hash(code_word) if code_word else None,
                 datetime.now().isoformat()))
            conn.commit()
        except sqlite3.IntegrityError:
            conn.close()
            return render_template("register.html",
                                   error="An account with that email already exists.")
        conn.close()
        return redirect("/login")
    return render_template("register.html")

@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        email    = request.form["email"].strip().lower()
        password = request.form["password"]
        conn = get_db()
        user = conn.execute(
            "SELECT id,name,password_hash FROM users WHERE email=?", (email,)
        ).fetchone()
        conn.close()
        if user and check_password_hash(user["password_hash"], password):
            session["user_id"] = user["id"]
            session["name"]    = user["name"]
            return redirect("/dashboard")
        return render_template("login.html", error="Invalid email or password.")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

# ── LOCATION / ANOMALY ENGINE ─────────────────────────────────────────────────

@app.route("/update_location", methods=["POST"])
def update_location():
    if "user_id" not in session:
        return jsonify({"status":"LOGIN_REQUIRED"}), 401
    try:
        lat = float(request.form["lat"])
        lon = float(request.form["lon"])
    except (KeyError, ValueError):
        return jsonify({"status":"ERROR","reason":"lat/lon missing or invalid"}), 400

    user_id = session["user_id"]
    now     = datetime.now()
    conn    = get_db()

    last_point = conn.execute(
        "SELECT latitude,longitude,speed_kmh,time FROM routes"
        " WHERE user_id=? ORDER BY id DESC LIMIT 1", (user_id,)).fetchone()

    speed_kmh   = 0.0
    gap_minutes = None
    if last_point:
        last_time   = datetime.fromisoformat(last_point["time"])
        gap_minutes = (now - last_time).total_seconds() / 60.0
        if gap_minutes > (2.0/60.0):
            dist_km   = haversine_km(lat,lon,last_point["latitude"],last_point["longitude"])
            speed_kmh = dist_km / (gap_minutes/60.0)

    # trip bookkeeping
    active_trip = get_active_trip(conn, user_id)
    if active_trip is None:
        trip_id     = start_trip(conn, user_id, lat, lon, now)
        active_trip = conn.execute("SELECT * FROM trips WHERE id=?", (trip_id,)).fetchone()
    elif gap_minutes is not None and gap_minutes > TRIP_GAP_MINUTES:
        finalize_trip(conn, active_trip)
        trip_id     = start_trip(conn, user_id, lat, lon, now)
        active_trip = conn.execute("SELECT * FROM trips WHERE id=?", (trip_id,)).fetchone()
    else:
        touch_trip(conn, active_trip["id"], lat, lon, now)
        trip_id = active_trip["id"]

    # risk zone proximity
    area_risk       = 0.0
    nearest_risk_km = 50.0
    for rz in conn.execute("SELECT latitude,longitude,risk_score FROM risk_zones").fetchall():
        d = haversine_km(lat,lon,rz["latitude"],rz["longitude"])
        if d < nearest_risk_km:
            nearest_risk_km = d
            area_risk = rz["risk_score"]

    # is_night
    is_night = 1 if (now.hour >= 22 or now.hour < 5) else 0

    # sudden_stop
    last_two   = conn.execute(
        "SELECT speed_kmh,time FROM routes WHERE user_id=? ORDER BY id DESC LIMIT 2",
        (user_id,)).fetchall()
    sudden_stop = 0
    if len(last_two) >= 2:
        prev_speed = last_two[1]["speed_kmh"] or 0.0
        gap_secs   = (now - datetime.fromisoformat(last_two[1]["time"])).total_seconds()
        if prev_speed >= 15 and speed_kmh < 3 and 0 < gap_secs <= 60:
            sudden_stop = 1

    start_zone   = zone_key(active_trip["start_lat"], active_trip["start_lon"])
    current_zone = zone_key(lat, lon)
    distance_from_start = haversine_km(lat, lon, active_trip["start_lat"], active_trip["start_lon"])
    total_prior_pings = conn.execute(
        "SELECT COUNT(*) c FROM routes WHERE user_id=?", (user_id,)
    ).fetchone()["c"]

    if total_prior_pings < ONBOARDING_GRACE_PINGS:
        ml_anomaly, ml_score = False, 0.0
        status, reason = "SAFE", "Still learning your travel patterns."
    elif active_trip["confirmed_safe"]:
        ml_anomaly, ml_score = False, 0.0
        status, reason = "SAFE", "You already confirmed this trip is safe."
    else:
        known_area = conn.execute(
            "SELECT 1 FROM verified_routes"
            " WHERE user_id=? AND ABS(latitude-?)<? AND ABS(longitude-?)<? LIMIT 1",
            (user_id,lat,KNOWN_AREA_RADIUS_DEG,lon,KNOWN_AREA_RADIUS_DEG)
        ).fetchone() is not None

        known_destinations      = known_destinations_for_start(conn, user_id, start_zone)
        has_any_pattern_history = conn.execute(
            "SELECT 1 FROM daily_patterns WHERE user_id=? AND start_zone=? LIMIT 1",
            (user_id, start_zone)).fetchone() is not None

        ml_anomaly, ml_score = compute_ml_score(
            lat, lon, now.hour, now.weekday(), speed_kmh,
            is_night=is_night, sudden_stop=sudden_stop,
            nearest_risk_km=nearest_risk_km, area_risk_score=area_risk)

        effective_radius = KNOWN_AREA_RADIUS_DEG * (0.5 if is_night else 1.0)
        known_area_eff = known_area or conn.execute(
            "SELECT 1 FROM verified_routes"
            " WHERE user_id=? AND ABS(latitude-?)<? AND ABS(longitude-?)<? LIMIT 1",
            (user_id,lat,effective_radius,lon,effective_radius)
        ).fetchone() is not None

        status = "SAFE"; reason = None

        if (current_zone != start_zone and has_any_pattern_history
                and known_destinations
                and current_zone not in known_destinations
                and not known_area_eff
                and distance_from_start >= MIN_MOVEMENT_KM):
            status = "NEW_ROUTE"
            reason = "This trip is heading somewhere different from your usual routes."
        elif current_zone != start_zone and not known_area_eff and not has_any_pattern_history and distance_from_start >= MIN_MOVEMENT_KM:
            status = "NEW_ROUTE"
            reason = "You haven't been tracked in this area before."

        if sudden_stop and nearest_risk_km < 0.5 and area_risk >= 0.65:
            status = "NEW_ROUTE"
            reason = f"Sudden stop near a high-risk area (risk score {area_risk:.2f})."

        if is_night and nearest_risk_km < 0.3 and area_risk >= 0.75 and status == "SAFE":
            status = "NEW_ROUTE"
            reason = f"You are in a high-risk area late at night (risk score {area_risk:.2f})."

        if status == "SAFE" and ml_anomaly and distance_from_start >= MIN_MOVEMENT_KM:
            status = "NEW_ROUTE"
            reason = "Unusual movement pattern flagged by the anomaly model."

    if status == "NEW_ROUTE":
        conn.execute(
            "INSERT INTO anomalies(user_id,latitude,longitude,anomaly_score,reason,time)"
            " VALUES(?,?,?,?,?,?)",
            (user_id,lat,lon,ml_score,reason,now.isoformat()))

    conn.execute(
        "INSERT INTO routes(user_id,trip_id,latitude,longitude,speed_kmh,time)"
        " VALUES(?,?,?,?,?,?)",
        (user_id,trip_id,lat,lon,speed_kmh,now.isoformat()))
    conn.commit()
    conn.close()

    return jsonify({
        "status":          status,
        "reason":          reason,
        "speed_kmh":       round(speed_kmh,1),
        "ml_score":        round(ml_score,3),
        "ml_ready":        ML_READY,
        "area_risk":       round(area_risk,2),
        "nearest_risk_km": round(nearest_risk_km,2),
        "is_night":        is_night,
        "sudden_stop":     sudden_stop,
    })

@app.route("/confirm_safe", methods=["POST"])
def confirm_safe():
    if "user_id" not in session:
        return jsonify({"message":"Login required"}), 401
    try:
        lat = float(request.form["lat"])
        lon = float(request.form["lon"])
    except (KeyError,ValueError):
        return jsonify({"message":"lat/lon invalid"}), 400
    conn = get_db()
    conn.execute("INSERT INTO verified_routes(user_id,latitude,longitude) VALUES(?,?,?)",
                 (session["user_id"],lat,lon))
    active_trip = get_active_trip(conn, session["user_id"])
    if active_trip:
        increment_pattern(conn, session["user_id"],
                          zone_key(active_trip["start_lat"],active_trip["start_lon"]),
                          zone_key(lat,lon))
        conn.execute("UPDATE trips SET confirmed_safe=1 WHERE id=?", (active_trip["id"],))
    conn.commit()
    conn.close()
    return jsonify({"message":"Route verified and learned"})

# ── SOS / CODE WORD ───────────────────────────────────────────────────────────

@app.route("/sos", methods=["POST"])
def sos():
    if "user_id" not in session:
        return jsonify({"message":"Login required"}), 401
    try:
        lat = float(request.form.get("lat",0) or 0)
        lon = float(request.form.get("lon",0) or 0)
    except ValueError:
        lat = lon = 0.0
    reason = request.form.get("reason","Manual SOS")
    conn = get_db()
    contacts = conn.execute(
        "SELECT name, phone FROM contacts WHERE user_id=?", (session["user_id"],)
    ).fetchall()
    contact_names = [c["name"] for c in contacts]
    photo_bytes = None
    if "photo" in request.files:
        f = request.files["photo"]
        if f and f.filename:
            photo_bytes = f.read()
    insert_alert(conn, session["user_id"], lat, lon, reason, send_sms=True, photo_bytes=photo_bytes)
    conn.close()
    return jsonify({
        "message": "SOS Sent",
        "sms_sent": TWILIO_ENABLED,
        "email_sent": EMAIL_ENABLED,
        "contacts_notified": contact_names
    })

@app.route("/verify_code_word", methods=["POST"])
def verify_code_word():
    if "user_id" not in session:
        return jsonify({"match":False}), 401
    word = request.form.get("word","").strip().lower()
    conn = get_db()
    user = conn.execute("SELECT code_word_hash FROM users WHERE id=?",
                        (session["user_id"],)).fetchone()
    match = bool(user and user["code_word_hash"]
                 and check_password_hash(user["code_word_hash"], word))
    if match:
        try:
            lat = float(request.form.get("lat",0) or 0)
            lon = float(request.form.get("lon",0) or 0)
        except ValueError:
            lat = lon = 0.0
        photo_bytes = None
        if "photo" in request.files:
            f = request.files["photo"]
            if f and f.filename:
                photo_bytes = f.read()
        insert_alert(conn, session["user_id"], lat, lon,
                     "Secret code word triggered", send_sms=True, photo_bytes=photo_bytes)
    conn.close()
    return jsonify({"match": match})

@app.route("/resolve_alert/<int:id>")
def resolve_alert(id):
    if "user_id" not in session:
        return redirect("/login")
    conn = get_db()
    conn.execute("UPDATE alerts SET resolved=1 WHERE id=? AND user_id=?",
                 (id, session["user_id"]))
    conn.commit()
    conn.close()
    return redirect("/analytics")

@app.route("/risk_zones")
def risk_zones_api():
    conn = get_db()
    rows = conn.execute("SELECT latitude,longitude,risk_score FROM risk_zones").fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

if __name__ == "__main__":
    app.run(debug=True)
