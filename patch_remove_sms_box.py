# patch_remove_sms_box.py — removes the "SMS alerts are off/active" box from Settings
# Run from the project folder:  python patch_remove_sms_box.py
import os
import re

PATHS = ["templates/settings.html", "_theme_backup/templates/settings.html"]
PATTERN = re.compile(
    r"[ \t]*\{%-?\s*if\s+twilio_enabled\s*-?%\}.*?\{%-?\s*endif\s*-?%\}[ \t]*\n?",
    re.S)

done = False
for path in PATHS:
    if not os.path.exists(path):
        continue
    with open(path, "r", encoding="utf-8") as f:
        html = f.read()
    new, n = PATTERN.subn("", html, count=1)
    if n and "SMS alerts" not in new:
        with open(path, "w", encoding="utf-8") as f:
            f.write(new)
        print("Removed SMS box from", path)
        done = True
    elif "twilio_enabled" not in html and "SMS alerts" not in html:
        print("Already removed in", path)
        done = True
    else:
        print("WARNING: could not cleanly remove it from", path, "- send me a screenshot")

if not done:
    print("ERROR: templates/settings.html not found - run this from the project folder")
else:
    print("PATCH COMPLETE - restart the app and press Ctrl+Shift+R")