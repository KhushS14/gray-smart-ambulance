# Read the app.py file with UTF-8 encoding
with open('api/app.py', 'r', encoding='utf-8') as f:  # ← Added encoding='utf-8'
    content = f.read()

print("Checking api/app.py for required updates...")
print("=" * 60)

checks = [
    ("Logging setup", "import logging" in content),
    ("Logger created", "logger = logging.getLogger" in content),
    ("Safety triggered variable", "safety_triggered = False" in content),
    ("Enhanced safety comment", "ENHANCED SAFETY OVERRIDE" in content),
    ("Safety override in response", '"safety_override": safety_triggered' in content),
    ("SpO2 < 90 check", "if min_spo2 < 90:" in content),
    ("HR > 140 check", "if max_hr > 140:" in content),
]

all_passed = True
for name, passed in checks:
    status = "✅" if passed else "❌"
    print(f"{status} {name}")
    if not passed:
        all_passed = False

print("=" * 60)
if all_passed:
    print("✅ All checks passed! Code is correctly updated.")
    print("\nNext step: Restart the server with:")
    print("   uvicorn api.app:app --reload")
else:
    print("❌ Code is NOT updated correctly!")
    print("\nYou need to replace api/app.py with the updated code.")