import json
from pathlib import Path
import requests

API_URL = "http://127.0.0.1:8000"

# 1. Health Check
print("=" * 65)
print("1. CHECKING API HEALTH (/ping)...")
try:
    res = requests.get(f"{API_URL}/ping")
    print(json.dumps(res.json(), indent=2))
except Exception as e:
    print(f"Failed to connect to API: {e}")
    print("Ensure the API server is running before executing this test.")
    exit(1)

# 2. Test predictions with actual dataset images
test_folders = [
    Path("Plant Disease Dataset/Tomato/Early blight"),
    Path("Plant Disease Dataset/Potato/Late blight"),
    Path("Plant Disease Dataset/Cotton/Leaf Curl Virus"),
    Path("Plant Disease Dataset/Corn/Common Rust"),
    Path("Plant Disease Dataset/Wheat/Yellow Rust"),
    Path("Plant Disease Dataset/Sugarcane/Red Rot"),
]

print("\n" + "=" * 65)
print("2. TESTING PREDICTIONS ACROSS DIFFERENT CROPS (/predict)...")
print("=" * 65)

for folder in test_folders:
    if not folder.exists():
        continue

    images = list(folder.glob("*.jpg")) + list(folder.glob("*.JPG")) + list(folder.glob("*.png"))
    if not images:
        continue

    sample_img = images[0]
    expected_class = f"{folder.parent.name}___{folder.name}"

    with open(sample_img, "rb") as f:
        res = requests.post(f"{API_URL}/predict", files={"file": (sample_img.name, f, "image/jpeg")})

    if res.status_code == 200:
        data = res.json()
        print(f"\nImage File:    {sample_img.name}")
        print(f"Expected:      {expected_class}")
        print(f"Predicted:     {data['prediction']}")
        print(f"Confidence:    {data['confidence']}%")
        print(f"Is Confident:  {data['is_confident']}")
        print(f"Gujarati Name: {data['details'].get('name_gu', 'N/A')} ({data['details'].get('plant_gu', 'N/A')})")
        print(f"Top 3:         {data['top_3']}")
    else:
        print(f"Error for {sample_img.name}: {res.status_code} - {res.text}")

print("\n" + "=" * 65)
print("ALL TESTS COMPLETED SUCCESSFULLY!")
print("=" * 65)