
import json
import os
from collections import Counter

# File paths
eu_file = r"c:\Users\kcs\Documents\kcs_hs_chatbot\knowledge\hs_classification_data_eu.json"
us_file = r"c:\Users\kcs\Documents\kcs_hs_chatbot\knowledge\hs_classification_data_us.json"

def get_hs_code_field(data):
    if not data:
        return None
    item = data[0]
    possible_keys = ['hs_code', 'hscode', 'classification', 'taric_code', 'hts_code']
    for key in possible_keys:
        if key in item:
            return key
    # If not found, look for keys containing 'code'
    for key in item.keys():
        if 'code' in key.lower():
            return key
    return None

def analyze_file(filepath):
    print(f"Analyzing {os.path.basename(filepath)}...")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return Counter(), None

    hs_key = get_hs_code_field(data)
    print(f"  HS Code Key: {hs_key}")
    
    codes = []
    descriptions = {} # Store a description for each code to help us understand what it is
    
    desc_key = 'description' if 'description' in data[0] else 'product_description'
    if desc_key not in data[0]:
         for key in data[0].keys():
            if 'desc' in key.lower():
                desc_key = key
                break
    print(f"  Description Key: {desc_key}")

    for item in data:
        code = item.get(hs_key)
        if code:
            # Normalize code: remove dots, spaces, take first 6 digits
            clean_code = str(code).replace('.', '').replace(' ', '').strip()[:6]
            if len(clean_code) >= 4:
                codes.append(clean_code)
                if clean_code not in descriptions and item.get(desc_key):
                    descriptions[clean_code] = item.get(desc_key)

    return Counter(codes), descriptions

eu_counts, eu_descs = analyze_file(eu_file)
us_counts, us_descs = analyze_file(us_file)


# Find intersection considering both frequencies
common_codes = set(eu_counts.keys()) & set(us_counts.keys())
print(f"\nCommon Codes Count: {len(common_codes)}")

combined_score = {}
for code in common_codes:
    combined_score[code] = eu_counts[code] + us_counts[code]

print("\nTop 10 Common Codes (by combined frequency):")
sorted_common = sorted(combined_score.items(), key=lambda x: x[1], reverse=True)
for code, score in sorted_common[:15]:
     print(f"Code: {code}, Combined: {score} (EU: {eu_counts[code]}, US: {us_counts[code]})")
     # Get descriptions from both to give a better idea
     desc_eu = eu_descs.get(code, "N/A")
     desc_us = us_descs.get(code, "N/A")
     print(f"  EU Desc: {desc_eu[:60]}...")
     print(f"  US Desc: {desc_us[:60]}...")
     print("-" * 20)

