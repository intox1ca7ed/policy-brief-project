import requests
import json
import pandas as pd
import logging
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=r"C:\Users\82154\Desktop\Digital public spheres\policy-brief-project\petition_analysis\petition_scraper\scraper.log"
)

# Step 1: Fetch and process a single JSON file
def fetch_and_process_json(petition_id):
    url = f"https://petition.parliament.uk/petitions/{petition_id}.json"
    try:
        response = requests.get(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36'
        })
        response.raise_for_status()
        data = response.json()
        
        petition_data = data.get('data', {}).get('attributes', {})
        
        # Extract fields
        petition_id = str(data.get('data', {}).get('id', petition_id))
        title = petition_data.get('action', 'Unknown')
        signature_count = petition_data.get('signature_count', 0)
        status = 'Responded' if petition_data.get('government_response') else 'Unknown'
        date = petition_data.get('created_at', 'Unknown')  # Use response date if available
        text = petition_data.get('background', '') or petition_data.get('additional_details', '') or '[Paste petition text here]'
        response_text = petition_data.get('government_response', {}).get('details', '[Paste government response text here]')
        
        return {
            'petition_id': petition_id,
            'title': title,
            'text': text,
            'signature_count': signature_count,
            'response_text': response_text,
            'status': status,
            'date': date
        }
    except Exception as e:
        logging.error(f"Error fetching JSON for petition {petition_id}: {e}")
        return None

# Step 2: Process all petition IDs
def process_petitions(petition_ids):
    petitions = []
    for petition_id in petition_ids:
        petition_data = fetch_and_process_json(petition_id)
        if petition_data:
            petitions.append(petition_data)
    return petitions

# Step 3: Save to CSV
def save_to_csv(petitions, filename=r"C:\Users\82154\Desktop\Digital public spheres\policy-brief-project\petition_analysis\petition_scraper\petitions_with_response_json.csv"):
    df = pd.DataFrame(petitions)
    df.to_csv(filename, index=False, encoding='utf-8')
    logging.info(f"Data saved to {filename}")

# Main execution
def main():
    # Petition IDs
    petition_ids = ['704793', '710486', '705772', '700161', '716515','700041', '701850', '718406', '716157', '716686', '715292', '717065','700029', '721547', '702424', '700168', '700292', '701064', '701159', '701517', '701838', '702341', '702538', '703284', '706884', '710067', '711976', '712763', '713128', '713714' ]
    
    logging.info("Starting JSON fetching on May 16, 2025")
    petitions = process_petitions(petition_ids)
    
    if petitions:
        save_to_csv(petitions)
        logging.info(f"Processed {len(petitions)} petitions")
    else:
        logging.warning("No petitions processed")

if __name__ == "__main__":
    main()