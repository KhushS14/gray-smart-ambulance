import json
import urllib.request

# Simplest possible test
test_data = {
    "time_sec": [0, 1, 2],
    "heart_rate": [160.0, 160.0, 160.0],  # Critical HR
    "spo2": [70.0, 70.0, 70.0],           # Critical SpO2
    "sbp": [80.0, 80.0, 80.0],            # Critical SBP
    "dbp": [50.0, 50.0, 50.0],
    "motion": [0.05, 0.05, 0.05]
}

print("Sending simple 3-second critical patient test...")
data = json.dumps(test_data).encode('utf-8')
req = urllib.request.Request(
    "http://localhost:8000/predict",
    data=data,
    headers={'Content-Type': 'application/json'}
)

try:
    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read().decode('utf-8'))
    print("\nResponse:")
    print(json.dumps(result, indent=2))
    
    print(f"\nRisk Score: {result.get('risk_score', 0.0)}")
    print(f"Safety Override: {result.get('safety_override', False)}")
    
except Exception as e:
    print(f"Error: {e}")