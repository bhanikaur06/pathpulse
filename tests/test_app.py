"""
Automated tests for Path Pulse. Uses only the Python standard library
(unittest) — no extra installs needed. Run with:

    python -m unittest tests.test_app -v

Uses a fresh temporary database per test — never touches your real
routes.db. ML scoring is forced off for the trip/pattern tests so those
are deterministic regardless of whatever model is on disk; the ML
fallback behaviour itself is tested separately.
"""

import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app as app_module


class PathPulseTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp_dir, "test_routes.db")
        self._orig_db_path = app_module.DB_PATH
        self._orig_compute_ml_score = app_module.compute_ml_score
        self._orig_ml_ready = app_module.ML_READY

        app_module.DB_PATH = self.db_path
        app_module.init_db()
        app_module.app.config["TESTING"] = True
        self.client = app_module.app.test_client()

    def tearDown(self):
        app_module.DB_PATH = self._orig_db_path
        app_module.compute_ml_score = self._orig_compute_ml_score
        app_module.ML_READY = self._orig_ml_ready
        try:
            os.remove(self.db_path)
        except OSError:
            pass

    def disable_ml(self):
        """Force the ML signal off so trip/pattern tests are deterministic
        regardless of whatever model happens to be on disk."""
        app_module.compute_ml_score = lambda *a, **k: (False, 0.0)

    def register_and_login(self, email="test@example.com", password="pass123", code_word="sunflower"):
        self.client.post("/register", data={
            "name": "Test User", "email": email, "password": password, "code_word": code_word,
        })
        self.client.post("/login", data={"email": email, "password": password})


# ---------------- auth ---------------- #

class TestAuth(PathPulseTestCase):
    def test_register_and_login(self):
        r = self.client.post("/register", data={
            "name": "Alice", "email": "alice@example.com", "password": "secret123", "code_word": "owl",
        }, follow_redirects=True)
        self.assertEqual(r.status_code, 200)

        r = self.client.post("/login", data={"email": "alice@example.com", "password": "secret123"},
                              follow_redirects=True)
        self.assertEqual(r.status_code, 200)

    def test_duplicate_email_rejected(self):
        self.client.post("/register", data={"name": "A", "email": "dupe@example.com", "password": "secret123"})
        r = self.client.post("/register", data={"name": "B", "email": "dupe@example.com", "password": "secret123"})
        self.assertIn(b"already exists", r.data)

    def test_short_password_rejected(self):
        r = self.client.post("/register", data={"name": "A", "email": "a@example.com", "password": "123"})
        self.assertIn(b"at least 6 characters", r.data)

    def test_wrong_password_rejected(self):
        self.client.post("/register", data={"name": "A", "email": "a@example.com", "password": "secret123"})
        r = self.client.post("/login", data={"email": "a@example.com", "password": "wrongpass"})
        self.assertIn(b"Invalid email or password", r.data)

    def test_passwords_are_hashed_not_plaintext(self):
        self.client.post("/register", data={"name": "A", "email": "a@example.com", "password": "secret123"})
        conn = app_module.get_db()
        row = conn.execute("SELECT password_hash FROM users WHERE email='a@example.com'").fetchone()
        conn.close()
        self.assertNotEqual(row["password_hash"], "secret123")
        self.assertTrue(row["password_hash"].startswith(("pbkdf2:", "scrypt:")))

    def test_dashboard_requires_login(self):
        r = self.client.get("/dashboard", follow_redirects=False)
        self.assertEqual(r.status_code, 302)
        self.assertIn("/login", r.headers["Location"])


# ---------------- contacts ---------------- #

class TestContacts(PathPulseTestCase):
    def test_add_and_delete_contact(self):
        self.register_and_login()
        r = self.client.post("/contacts", data={"name": "Mom", "phone": "12345", "relationship": "Mother"},
                              follow_redirects=True)
        self.assertIn(b"Mom", r.data)

        conn = app_module.get_db()
        contact_id = conn.execute("SELECT id FROM contacts WHERE name='Mom'").fetchone()["id"]
        conn.close()

        r = self.client.get(f"/delete_contact/{contact_id}", follow_redirects=True)
        self.assertNotIn(b"Mom", r.data)


# ---------------- location / trip / pattern logic ---------------- #

class TestLocationAndPatterns(PathPulseTestCase):
    def test_cold_start_first_ping_is_safe(self):
        """The very first ping of any trip IS the trip's start point — you
        can't have deviated from your own starting point, so this should
        always be SAFE, even for a brand-new user with zero history."""
        self.disable_ml()
        self.register_and_login()
        r = self.client.post("/update_location", data={"lat": 28.4595, "lon": 77.0266}).get_json()
        self.assertEqual(r["status"], "SAFE")

    def test_cold_start_moving_with_no_history_is_new_route(self):
        """Once a brand-new user actually moves somewhere, with zero travel
        history Path Pulse correctly has nothing to compare against."""
        self.disable_ml()
        self.register_and_login()
        self.client.post("/update_location", data={"lat": 28.4595, "lon": 77.0266})
        r = self.client.post("/update_location", data={"lat": 28.9000, "lon": 77.6000}).get_json()
        self.assertEqual(r["status"], "NEW_ROUTE")

    def test_bad_lat_lon_returns_400(self):
        self.register_and_login()
        r = self.client.post("/update_location", data={"lat": "not-a-number", "lon": "77"})
        self.assertEqual(r.status_code, 400)
        self.assertEqual(r.get_json()["status"], "ERROR")

    def test_update_location_requires_login(self):
        r = self.client.post("/update_location", data={"lat": 1, "lon": 1})
        self.assertEqual(r.status_code, 401)

    def test_confirm_safe_suppresses_rest_of_trip(self):
        self.disable_ml()
        self.register_and_login()
        home = (28.4595, 77.0266)
        far = (28.9000, 77.6000)

        self.client.post("/update_location", data={"lat": home[0], "lon": home[1]})
        r = self.client.post("/update_location", data={"lat": far[0], "lon": far[1]}).get_json()
        self.assertEqual(r["status"], "NEW_ROUTE")

        self.client.post("/confirm_safe", data={"lat": far[0], "lon": far[1]})

        r = self.client.post(
            "/update_location", data={"lat": far[0] + 0.01, "lon": far[1] + 0.01}
        ).get_json()
        self.assertEqual(r["status"], "SAFE")

    def test_pattern_becomes_normal_after_three_trips(self):
        self.disable_ml()
        self.register_and_login()
        home = (28.4595, 77.0266)
        college = (28.4700, 77.0400)

        def ping(lat, lon):
            return self.client.post("/update_location", data={"lat": lat, "lon": lon}).get_json()

        def backdate_last_route(minutes):
            conn = app_module.get_db()
            conn.execute(
                "UPDATE routes SET time=? WHERE id = (SELECT MAX(id) FROM routes)",
                ((datetime.now() - timedelta(minutes=minutes)).isoformat(),)
            )
            conn.commit()
            conn.close()

        statuses = []
        for _ in range(4):
            ping(*home)
            backdate_last_route(2)
            statuses.append(ping(*college)["status"])
            backdate_last_route(20)

        self.assertEqual(statuses[-1], "SAFE")

        conn = app_module.get_db()
        pattern = conn.execute("SELECT * FROM daily_patterns").fetchone()
        conn.close()
        self.assertEqual(pattern["status"], "NORMAL")
        self.assertGreaterEqual(pattern["trip_count"], 3)

    def test_stationary_trip_does_not_pollute_patterns(self):
        self.disable_ml()
        self.register_and_login()
        home = (28.4595, 77.0266)

        self.client.post("/update_location", data={"lat": home[0], "lon": home[1]})

        conn = app_module.get_db()
        conn.execute(
            "UPDATE routes SET time=? WHERE id = (SELECT MAX(id) FROM routes)",
            ((datetime.now() - timedelta(minutes=20)).isoformat(),)
        )
        conn.commit()
        conn.close()

        self.client.post("/update_location", data={"lat": home[0], "lon": home[1]})

        conn = app_module.get_db()
        patterns = conn.execute("SELECT * FROM daily_patterns").fetchall()
        conn.close()
        self.assertEqual(len(patterns), 0)


# ---------------- SOS / code word ---------------- #

class TestSosAndCodeWord(PathPulseTestCase):
    def test_manual_sos_creates_alert(self):
        self.register_and_login()
        r = self.client.post("/sos", data={"lat": 1.0, "lon": 1.0, "reason": "Manual SOS"}).get_json()
        self.assertEqual(r["message"], "SOS Sent")

        conn = app_module.get_db()
        count = conn.execute("SELECT COUNT(*) c FROM alerts").fetchone()["c"]
        conn.close()
        self.assertEqual(count, 1)

    def test_correct_code_word_triggers_silent_alert(self):
        self.register_and_login(code_word="sunflower")
        r = self.client.post("/verify_code_word", data={"word": "sunflower", "lat": 1, "lon": 1}).get_json()
        self.assertTrue(r["match"])

        conn = app_module.get_db()
        count = conn.execute("SELECT COUNT(*) c FROM alerts").fetchone()["c"]
        conn.close()
        self.assertEqual(count, 1)

    def test_wrong_code_word_does_not_trigger_alert(self):
        self.register_and_login(code_word="sunflower")
        r = self.client.post("/verify_code_word", data={"word": "wrongword", "lat": 1, "lon": 1}).get_json()
        self.assertFalse(r["match"])

        conn = app_module.get_db()
        count = conn.execute("SELECT COUNT(*) c FROM alerts").fetchone()["c"]
        conn.close()
        self.assertEqual(count, 0)

    def test_sos_requires_login(self):
        r = self.client.post("/sos", data={"lat": 1, "lon": 1})
        self.assertEqual(r.status_code, 401)


# ---------------- ML scoring ---------------- #

class TestMlScoring(PathPulseTestCase):
    def test_compute_ml_score_returns_safe_defaults_when_model_absent(self):
        app_module.ML_READY = False
        is_anomaly, score = app_module.compute_ml_score(28.46, 77.03, 8, 1, 5.0)
        self.assertFalse(is_anomaly)
        self.assertEqual(score, 0.0)


# ---------------- pages render without error ---------------- #

class TestPagesLoad(PathPulseTestCase):
    def test_public_pages_load(self):
        for path in ["/", "/login", "/register"]:
            with self.subTest(path=path):
                self.assertEqual(self.client.get(path).status_code, 200)

    def test_protected_pages_load_when_logged_in(self):
        self.register_and_login()
        for path in ["/dashboard", "/history", "/analytics", "/contacts"]:
            with self.subTest(path=path):
                self.assertEqual(self.client.get(path).status_code, 200)


if __name__ == "__main__":
    unittest.main()
