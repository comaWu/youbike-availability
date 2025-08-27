import requests, json
url = "https://data.ntpc.gov.tw/api/datasets/010e5b15-3823-4b20-b401-b1cf000550c5/json"
r = requests.get(url, timeout=15); r.raise_for_status()
data = r.json()
print("rows:", len(data))
print("sample keys:", list(data[0].keys())[:12])
print(json.dumps(data[0], ensure_ascii=False, indent=2))