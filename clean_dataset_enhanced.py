#!/usr/bin/env python3
"""
Enhanced Dataset Cleaning Script
Features:
1. Removes specified columns
2. Adds funding-based success criteria (>=100M)
3. Quantifies category_list and technology_stack for ML
"""

import csv
import json
import re
from collections import Counter
from typing import Dict, List, Set, Tuple

# Top categories based on startup ecosystem analysis
TOP_CATEGORIES = [
    'software', 'mobile', 'e-commerce', 'saas', 'health care',
    'finance', 'education', 'analytics', 'artificial intelligence',
    'enterprise software', 'biotechnology', 'marketplace', 'social media',
    'fintech', 'gaming', 'advertising', 'cloud computing', 'internet',
    'apps', 'medical', 'real estate', 'retail', 'social network'
]

# Top technologies based on modern tech stack
TOP_TECHNOLOGIES = [
    'artificial intelligence', 'machine learning', 'deep learning',
    'blockchain', 'cloud computing', 'mobile app', 'saas', 'api',
    'data science', 'big data', 'analytics', 'iot',
    'augmented reality', 'virtual reality', 'computer vision',
    'natural language processing', 'react', 'angular', 'python', 'java'
]


def parse_funding_amount(funding_str: str) -> float:
    """
    Parse funding string to numeric value
    Handles various formats: '1000000', '1M', '1.5M', '$1M', etc.
    """
    if not funding_str or funding_str == '-' or funding_str.strip() == '':
        return 0.0

    # Remove currency symbols and whitespace
    funding_str = str(funding_str).strip().replace('$', '').replace(',', '').upper()

    try:
        # Handle M (millions) and B (billions)
        if 'B' in funding_str:
            number = float(funding_str.replace('B', ''))
            return number * 1_000_000_000
        elif 'M' in funding_str:
            number = float(funding_str.replace('M', ''))
            return number * 1_000_000
        elif 'K' in funding_str:
            number = float(funding_str.replace('K', ''))
            return number * 1_000
        else:
            return float(funding_str)
    except ValueError:
        return 0.0


def determine_success(status: str, funding_amount: float) -> bool:
    """
    Determine if company is successful based on:
    1. Status (IPO or Acquired)
    2. Funding >= $100M (indicates unicorn potential)
    """
    # Original success criteria (IPO or Acquired)
    status_success = status.lower() in ['ipo', 'acquired'] if status else False

    # New funding-based criteria (>= $100M)
    funding_success = funding_amount >= 100_000_000

    # Company is successful if either condition is met
    return status_success or funding_success


def quantify_categories(category_str: str) -> Dict[str, any]:
    """
    Quantify category_list for ML models
    Returns:
    - category_count: Number of categories
    - category_diversity_score: Diversity metric (1-10)
    - Binary features for each top category
    - category_rarity_score: Average rarity score
    """
    features = {}

    if not category_str or category_str.strip() == '':
        features['category_count'] = 0
        features['category_diversity_score'] = 0
        features['category_rarity_score'] = 0.0

        # All top categories set to 0
        for cat in TOP_CATEGORIES:
            features[f'cat_{cat.replace(" ", "_").lower()}'] = 0

        return features

    # Split and clean categories
    categories = [c.strip().lower() for c in category_str.split('|') if c.strip()]

    # Count feature
    features['category_count'] = len(categories)

    # Diversity score (normalized, capped at 10)
    features['category_diversity_score'] = min(len(categories), 10)

    # Binary features for top categories
    for top_cat in TOP_CATEGORIES:
        # Check if any category contains this keyword
        match = any(top_cat in cat for cat in categories)
        features[f'cat_{top_cat.replace(" ", "_").lower()}'] = 1 if match else 0

    # Rarity score (inverse of common categories)
    # More niche/specific categories = higher rarity
    common_categories = {'software', 'internet', 'mobile', 'web'}
    niche_count = sum(1 for cat in categories if cat not in common_categories)
    features['category_rarity_score'] = round(niche_count / max(len(categories), 1), 2)

    return features


def quantify_technologies(tech_str: str) -> Dict[str, any]:
    """
    Quantify technology_stack for ML models
    Returns:
    - tech_count: Number of technologies mentioned
    - tech_modernity_score: Score based on modern tech (1-10)
    - Binary features for each top technology
    - has_ai_ml: Whether uses AI/ML
    - has_cloud: Whether uses cloud technologies
    """
    features = {}

    if not tech_str or tech_str.strip() == '' or tech_str.lower() == 'unknown':
        features['tech_count'] = 0
        features['tech_modernity_score'] = 0
        features['has_ai_ml'] = 0
        features['has_cloud'] = 0

        # All top technologies set to 0
        for tech in TOP_TECHNOLOGIES:
            features[f'tech_{tech.replace(" ", "_").lower()}'] = 0

        return features

    # Clean and split technologies
    tech_str_lower = tech_str.lower()
    technologies = [t.strip() for t in tech_str.split(',') if t.strip()]

    # Count feature
    features['tech_count'] = len(technologies)

    # Binary features for top technologies
    for top_tech in TOP_TECHNOLOGIES:
        match = top_tech in tech_str_lower
        features[f'tech_{top_tech.replace(" ", "_").lower()}'] = 1 if match else 0

    # AI/ML detection
    ai_keywords = ['artificial intelligence', 'machine learning', 'deep learning', 'ai', 'ml', 'neural']
    features['has_ai_ml'] = 1 if any(kw in tech_str_lower for kw in ai_keywords) else 0

    # Cloud detection
    cloud_keywords = ['cloud', 'aws', 'azure', 'gcp', 'kubernetes', 'docker']
    features['has_cloud'] = 1 if any(kw in tech_str_lower for kw in cloud_keywords) else 0

    # Modernity score (based on cutting-edge technologies)
    modern_tech = ['ai', 'machine learning', 'blockchain', 'kubernetes', 'react',
                   'cloud', 'microservices', 'docker', 'deep learning']
    modern_count = sum(1 for tech in modern_tech if tech in tech_str_lower)
    features['tech_modernity_score'] = min(modern_count, 10)

    return features


def clean_dataset():
    print("=" * 90)
    print("ENHANCED DATASET CLEANING - WITH ML QUANTIFICATION")
    print("=" * 90)
    print()

    # Define features to keep (AFTER REMOVAL)
    # Removed: permalink, homepage_url, country_code, state_code, region,
    #          funding_rounds, founded_at, first_funding_at, last_funding_at,
    #          founder_count, city
    original_features = [
        'name',
        'category_list',
        'funding_total_usd',
        'status',
    ]

    # Scraped features to KEEP
    scraped_features_to_keep = [
        # Content Analysis
        'company_description',
        'detected_language',

        # Business Intelligence
        'business_model_clarity',
        'technology_stack',

        # Founder Signals (removed founder_count as per requirements)
        'founder_education_quality',
        'founder_technical_background',
        'founder_business_background',
        'founder_previous_companies',

        # Target Variable
        'success'
    ]

    # Features being REMOVED
    removed_features = [
        # Per user request
        'permalink',
        'homepage_url',
        'country_code',
        'state_code',
        'region',
        'funding_rounds',
        'founded_at',
        'first_funding_at',
        'last_funding_at',
        'founder_count',
        'city',

        # Website Accessibility
        'website_accessible',
        'website_status_code',
        'website_error',

        # Content Analysis (redundant/removed)
        'word_count',
        'content_length',

        # Business Intelligence (redundant)
        'technology_count',

        # Founder Signals (sparse/unusable)
        'founder_names',

        # Business Metrics (all too sparse)
        'customer_mentions_count',
        'customer_mentions',
        'revenue_mentions_count',
        'revenue_mentions',
        'growth_mentions_count',
        'growth_mentions',
        'team_size_count',
        'team_size',
        'partnership_mentions',
        'award_mentions'
    ]

    print(f"📊 FEATURE SUMMARY:")
    print(f"   Original features kept: {len(original_features)}")
    print(f"   Scraped features kept: {len(scraped_features_to_keep)}")
    print(f"   Features being removed: {len(removed_features)}")
    print()

    print("✅ KEEPING THESE FEATURES:\n")
    all_keep = original_features + scraped_features_to_keep
    for i, feat in enumerate(all_keep, 1):
        print(f"   {i:2}. {feat}")
    print()

    print("❌ REMOVING THESE FEATURES:\n")
    for i, feat in enumerate(removed_features, 1):
        print(f"   {i:2}. {feat}")
    print()

    # Load original dataset
    print("📂 Loading enhanced_dataset_v3.csv...")
    data = []
    with open('output.csv', 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        original_columns = reader.fieldnames
        for row in reader:
            data.append(row)

    print(f"   Loaded {len(data)} rows")
    print(f"   Original columns: {len(original_columns)}")
    print()

    # Process and enhance dataset
    print("🔧 Processing dataset with ML quantification...")
    print()

    cleaned_data = []
    success_from_status = 0
    success_from_funding = 0
    success_from_both = 0

    for idx, row in enumerate(data):
        cleaned_row = {}

        # Keep basic features
        for feature in original_features + scraped_features_to_keep:
            if feature in row:
                cleaned_row[feature] = row[feature]
            else:
                cleaned_row[feature] = ''

        # Parse funding amount
        funding_amount = parse_funding_amount(row.get('funding_total_usd', '0'))

        # Determine success with new criteria
        status = row.get('status', '')
        old_success = row.get('success', 'False') == 'True'
        new_success = determine_success(status, funding_amount)

        cleaned_row['success'] = str(new_success)

        # Track success sources
        status_success = status.lower() in ['ipo', 'acquired'] if status else False
        funding_success = funding_amount >= 100_000_000

        if status_success and funding_success:
            success_from_both += 1
        elif status_success:
            success_from_status += 1
        elif funding_success:
            success_from_funding += 1

        # Quantify categories
        category_features = quantify_categories(row.get('category_list', ''))
        cleaned_row.update(category_features)

        # Quantify technologies
        tech_features = quantify_technologies(row.get('technology_stack', ''))
        cleaned_row.update(tech_features)

        cleaned_data.append(cleaned_row)

        # Progress indicator
        if (idx + 1) % 100 == 0:
            print(f"   Processed {idx + 1}/{len(data)} rows...", end='\r')

    print(f"   Processed {len(data)}/{len(data)} rows... ✓")
    print()

    # Print success criteria analysis
    print("=" * 90)
    print("SUCCESS CRITERIA ANALYSIS")
    print("=" * 90)
    print()
    total_success = success_from_status + success_from_funding + success_from_both
    print(f"   Total successful companies: {total_success} ({total_success/len(data)*100:.1f}%)")
    print(f"   ├─ From status only (IPO/Acquired): {success_from_status}")
    print(f"   ├─ From funding only (≥$100M):      {success_from_funding}")
    print(f"   └─ From both criteria:               {success_from_both}")
    print()

    # Get all quantified feature names
    if cleaned_data:
        all_fieldnames = list(cleaned_data[0].keys())
    else:
        all_fieldnames = []

    # Save cleaned dataset
    output_file = 'cleaned_enhanced_dataset.csv'
    print(f"💾 Saving to {output_file}...")

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=all_fieldnames)
        writer.writeheader()
        writer.writerows(cleaned_data)

    print(f"   ✓ Saved {len(cleaned_data)} rows")
    print(f"   ✓ Total features: {len(all_fieldnames)}")
    print()

    # Feature breakdown
    print("=" * 90)
    print("QUANTIFIED FEATURES SUMMARY")
    print("=" * 90)
    print()

    category_features = [f for f in all_fieldnames if f.startswith('cat_') or 'category' in f]
    tech_features = [f for f in all_fieldnames if f.startswith('tech_') or f in ['has_ai_ml', 'has_cloud']]

    print(f"📊 Category Features ({len(category_features)}):")
    print(f"   - category_count")
    print(f"   - category_diversity_score")
    print(f"   - category_rarity_score")
    print(f"   - {len([f for f in category_features if f.startswith('cat_')])} binary category indicators")
    print()

    print(f"💻 Technology Features ({len(tech_features)}):")
    print(f"   - tech_count")
    print(f"   - tech_modernity_score")
    print(f"   - has_ai_ml")
    print(f"   - has_cloud")
    print(f"   - {len([f for f in tech_features if f.startswith('tech_')])} binary technology indicators")
    print()

    # Calculate data quality stats
    print("=" * 90)
    print("DATA QUALITY ANALYSIS")
    print("=" * 90)
    print()

    # Category analysis
    has_categories = sum(1 for row in cleaned_data if row.get('category_count', '0') != '0')
    avg_categories = sum(int(row.get('category_count', 0)) for row in cleaned_data) / len(cleaned_data)

    print(f"📈 Category Analysis:")
    print(f"   Companies with categories: {has_categories}/{len(cleaned_data)} ({has_categories/len(cleaned_data)*100:.1f}%)")
    print(f"   Average categories per company: {avg_categories:.2f}")
    print()

    # Technology analysis
    has_tech = sum(1 for row in cleaned_data if row.get('tech_count', '0') != '0')
    avg_tech = sum(int(row.get('tech_count', 0)) for row in cleaned_data) / len(cleaned_data)

    print(f"💻 Technology Analysis:")
    print(f"   Companies with tech info: {has_tech}/{len(cleaned_data)} ({has_tech/len(cleaned_data)*100:.1f}%)")
    print(f"   Average technologies per company: {avg_tech:.2f}")
    print()

    # AI/ML and Cloud adoption
    has_ai = sum(1 for row in cleaned_data if row.get('has_ai_ml') == '1')
    has_cloud = sum(1 for row in cleaned_data if row.get('has_cloud') == '1')

    print(f"🚀 Modern Tech Adoption:")
    print(f"   Companies using AI/ML: {has_ai}/{len(cleaned_data)} ({has_ai/len(cleaned_data)*100:.1f}%)")
    print(f"   Companies using Cloud: {has_cloud}/{len(cleaned_data)} ({has_cloud/len(cleaned_data)*100:.1f}%)")
    print()

    print("=" * 90)
    print("✅ CLEANING COMPLETE")
    print("=" * 90)
    print()
    print(f"📄 Input:  enhanced_dataset_v3.csv ({len(original_columns)} columns)")
    print(f"📄 Output: {output_file} ({len(all_fieldnames)} columns)")
    print()
    print(f"🎯 Key Enhancements:")
    print(f"   ✓ Removed {len(removed_features)} low-value columns")
    print(f"   ✓ Added funding-based success criteria (≥$100M)")
    print(f"   ✓ Quantified {len(category_features)} category features")
    print(f"   ✓ Quantified {len(tech_features)} technology features")
    print(f"   ✓ Ready for ML model training!")
    print()
    print("=" * 90)


if __name__ == "__main__":
    clean_dataset()
