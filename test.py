import requests
import json

# API endpoint
API_URL = "http://localhost:8000/predict"

# Test case: Progressive patient deterioration
test_data = {
    "time_sec": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29],
    "heart_rate": [88,90,92,95,98,102,106,110,115,118,120,123,125,128,130,132,135,138,140,142,145,147,148,150,152,154,155,156,158,160],
    "spo2": [96,95,94,93,92,91,90,89,88,87,86,85,84,83,82,81,80,79,78,77,76,75,74,73,72,71,70,69,68,67],
    "sbp": [118,116,115,114,113,112,110,108,106,104,102,100,98,96,95,94,93,92,91,90,89,88,87,86,85,84,83,82,81,80],
    "dbp": [76,75,74,73,72,71,70,69,68,67,66,65,64,63,62,61,60,59,58,57,56,55,54,53,52,51,50,49,48,47],
    "motion": [0.05]*30
}

print("=" * 60)
print("🚑 TESTING SMART AMBULANCE EARLY WARNING API")
print("=" * 60)

print("\n📊 Test Scenario: Progressive Patient Deterioration")
print(f"   Initial → Final")
print(f"   HR:   {test_data['heart_rate'][0]} → {test_data['heart_rate'][-1]} bpm")
print(f"   SpO2: {test_data['spo2'][0]} → {test_data['spo2'][-1]}%")
print(f"   SBP:  {test_data['sbp'][0]} → {test_data['sbp'][-1]} mmHg")
print(f"   Duration: {len(test_data['time_sec'])} seconds")

print("\n🔄 Sending request to API...")

try:
    response = requests.post(API_URL, json=test_data)
    response.raise_for_status()
    
    result = response.json()
    
    print("\n" + "=" * 60)
    print("✅ API RESPONSE")
    print("=" * 60)
    print(json.dumps(result, indent=2))
    
    # Evaluate the response
    print("\n" + "=" * 60)
    print("🔍 EVALUATION")
    print("=" * 60)
    
    anomaly = result.get("anomaly", False)
    risk_score = result.get("risk_score", 0.0)
    confidence = result.get("confidence", 0.0)
    safety_override = result.get("safety_override", False)
    
    print(f"Anomaly Detected:  {'🔴 YES' if anomaly else '🟢 NO'}")
    print(f"Risk Score:        {risk_score:.1f}/100")
    print(f"Confidence:        {confidence:.1f}%")
    print(f"Safety Override:   {'✅ TRIGGERED' if safety_override else '❌ Not triggered'}")
    
    # Expected behavior
    print("\n" + "=" * 60)
    print("📋 EXPECTED BEHAVIOR")
    print("=" * 60)
    print("✓ Should detect anomaly:       YES (critical deterioration)")
    print("✓ Should have high risk score: ≥ 90.0 (multiple critical thresholds)")
    print("✓ Should trigger safety:       YES (SpO2=67, HR=160, SBP=80)")
    
    # Pass/Fail
    print("\n" + "=" * 60)
    if anomaly and risk_score >= 90.0 and safety_override:
        print("✅ TEST PASSED - System correctly identified critical patient")
    else:
        print("❌ TEST FAILED - Issues detected:")
        if not anomaly:
            print("   - Anomaly not detected")
        if risk_score < 90.0:
            print(f"   - Risk score too low ({risk_score:.1f} < 90.0)")
        if not safety_override:
            print("   - Safety override not triggered")
    print("=" * 60)
    
except requests.exceptions.ConnectionError:
    print("\n❌ ERROR: Could not connect to API")
    print("   Make sure the server is running: uvicorn api.app:app --reload")
except requests.exceptions.HTTPError as e:
    print(f"\n❌ HTTP ERROR: {e}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"\n❌ UNEXPECTED ERROR: {e}")