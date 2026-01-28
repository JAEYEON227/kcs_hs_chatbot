
import json
import glob
import os
from collections import Counter
import re

def analyze_domestic_data_frequency():
    # Load all domestic case files
    file_pattern = 'knowledge/HS분류사례*.json'
    files = glob.glob(file_pattern)
    
    all_cases = []
    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_cases.extend(data)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            
    print(f"Total domestic cases found: {len(all_cases)}")
    
    # Extract product names and keywords
    product_names = []
    descriptions = []
    
    for case in all_cases:
        if 'product_name' in case:
            product_names.append(case['product_name'])
        if 'description' in case:
            descriptions.append(case['description'])
            
    # Simple tokenization for frequency analysis
    def get_tokens(text_list):
        tokens = []
        for text in text_list:
            if not text: continue
            # Remove special chars and split
            cleaned = re.sub(r'[^\w\s가-힣]', ' ', text)
            words = cleaned.split()
            # Filter short words
            words = [w for w in words if len(w) >= 2]
            tokens.extend(words)
        return tokens

    name_tokens = get_tokens(product_names)
    desc_tokens = get_tokens(descriptions)
    
    name_counter = Counter(name_tokens)
    
    print("\n--- Top 20 Frequent Keywords in Product Names ---")
    for word, count in name_counter.most_common(20):
        print(f"{word}: {count}")

    # Check specifically for some common categories to find a good specific example
    categories = ['반도체', '화장품', '자동차', '디스플레이', '플라스틱', '영상', '카메라', '치즈', '조제']
    print("\n--- Frequency of Specific Categories ---")
    for cat in categories:
        count = 0
        for case in all_cases:
            text = (case.get('product_name') or '') + ' ' + (case.get('description') or '')
            if cat in text:
                count += 1
        print(f"{cat}: {count}")

if __name__ == "__main__":
    analyze_domestic_data_frequency()
