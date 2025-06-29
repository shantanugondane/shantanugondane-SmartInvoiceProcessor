import os
import json
from datetime import datetime
from supabase import create_client
from typing import Dict, Any
from PIL import Image
from dotenv import load_dotenv
import requests
import re
import base64

# Load environment variables
load_dotenv()

# Hugging Face Inference API configuration
HF_API_URL = "https://api-inference.huggingface.co/models/google/vit-base-patch16-224"
HF_API_KEY = os.getenv("HF_API_KEY")

# Debug information
print("Current working directory:", os.getcwd())
print("Environment variables loaded:", os.environ.get("HF_API_KEY") is not None)
print("API Key length:", len(HF_API_KEY) if HF_API_KEY else 0)

if not HF_API_KEY:
    raise ValueError(
        "Hugging Face API key not found. Please set HF_API_KEY in your .env file"
    )

# Headers for Hugging Face API calls
hf_headers = {
    "Authorization": f"Bearer {HF_API_KEY}"
    # Content-Type will be inferred for direct image uploads
}


def extract_info_from_text(text: str) -> Dict[str, str]:
    """
    Extracts invoice information from raw text using regex.
    This is a simplified example; real-world parsing might be more complex.
    """
    extracted = {"amount": "", "buyer": "", "seller": "", "date": ""}

    # Example patterns (these are very basic and will need refinement based on actual invoice text)
    # Amount: Look for a currency symbol followed by numbers (e.g., $123.45, €1.234,00)
    amount_match = re.search(
        r"(?:Total|Amount|Balance due|Invoice Total)[:\s]*[\$€£]?\s*([\d,]+\.?\d*)",
        text,
        re.IGNORECASE,
    )
    if amount_match:
        extracted["amount"] = amount_match.group(1).replace(",", "").strip()

    # Date: Look for common date formats
    date_match = re.search(
        r"Date[:\s]*(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4})", text, re.IGNORECASE
    )
    if not date_match:  # Try another common format
        date_match = re.search(
            r"(?:Invoice Date|Bill Date)[:\s]*(\w+\s+\d{1,2},\s+\d{4})",
            text,
            re.IGNORECASE,
        )
    if date_match:
        extracted["date"] = date_match.group(1).strip()

    # Buyer/Seller: This is highly dependent on text structure.
    # For now, let's keep it simple, might need more advanced NLP or fixed positions.
    # This part will be very generic and likely require manual adjustment for real invoices.
    # A more robust solution would involve named entity recognition or deeper parsing.

    # Placeholder for buyer:
    buyer_match = re.search(
        r"Bill To[:\s]*(.*?)(?:\n|$)", text, re.IGNORECASE | re.DOTALL
    )
    if buyer_match:
        extracted["buyer"] = (
            buyer_match.group(1).strip().split("\n")[0]
        )  # Take first line after "Bill To"

    # Placeholder for seller:
    seller_match = re.search(
        r"(?:From|Seller|Vendor)[:\s]*(.*?)(?:\n|$)", text, re.IGNORECASE | re.DOTALL
    )
    if seller_match:
        extracted["seller"] = seller_match.group(1).strip().split("\n")[0]

    return extracted


def process_invoice_image(image_path: str) -> Dict[str, Any]:
    """Process invoice image by first extracting text using Hugging Face API and then parsing it."""
    print(
        f"Extracting text from image using Hugging Face Inference API ({HF_API_URL} - Image Classification task for testing)..."
    )
    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        response = requests.post(HF_API_URL, headers=hf_headers, data=image_bytes)

        print(f"API Response Status Code: {response.status_code}")
        print(f"API Response Headers: {response.headers}")
        print(
            f"API Raw Response Text: {response.text[:500]}..."
        )  # Print first 500 chars of raw response

        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)

        # For image classification, response is typically a list of dicts with 'label' and 'score'
        api_response_content = response.json()

        # For this test, we'll just print the response to confirm API access
        print(
            f"Image Classification API Response: {json.dumps(api_response_content, indent=2)}"
        )

        # We don't proceed with text extraction or parsing for this diagnostic step
        return {}

    except (
        requests.exceptions.RequestException
    ) as e:  # Catch network/HTTP errors specifically
        print(f"Network or HTTP Error during API call: {type(e).__name__} - {e}")
        if hasattr(e, "response") and e.response is not None:
            print(f"API Error Response Status: {e.response.status_code}")
            print(f"API Error Response Text: {e.response.text[:500]}...")
        return {}
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {type(e).__name__} - {e}")
        print(f"Problematic raw response: {response.text[:500]}...")
        return {}
    except Exception as e:
        print(
            f"An unexpected generic error occurred during image processing: {type(e).__name__} - {e}"
        )
        return {}


def store_invoice_data(data: Dict[str, Any]) -> bool:
    """Store invoice data in Supabase."""
    try:
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        print("Supabase URL:", supabase_url)
        print("Supabase Key length:", len(supabase_key) if supabase_key else 0)
        if not supabase_url or not supabase_key:
            raise ValueError("Supabase credentials not found in environment variables")
        print("Initializing Supabase client...")
        supabase = create_client(supabase_url, supabase_key)
        invoice_data = {
            "amount": data.get("amount", ""),
            "buyer": data.get("buyer", ""),
            "seller": data.get("seller", ""),
            "date": data.get("date", ""),
            "processed_at": datetime.utcnow().isoformat(),
        }
        print("Attempting to store data:", json.dumps(invoice_data, indent=2))
        try:
            result = supabase.table("invoices").insert(invoice_data).execute()
            print("Supabase response:", result)
            if hasattr(result, "error") and result.error:
                print(f"Error storing data: {result.error}")
                return False
            return True
        except Exception as e:
            print(f"Error during Supabase insert: {str(e)}")
            return False
    except Exception as e:
        print(f"Error in store_invoice_data: {str(e)}")
        return False


def main():
    image_path = "invoice.png"
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found")
        return
    print("Processing invoice image via Hugging Face Inference API...")
    extracted_data = process_invoice_image(image_path)
    if not extracted_data or all(value == "" for value in extracted_data.values()):
        print("Failed to extract data from invoice or no data extracted.")
        return
    print("\nExtracted data:", json.dumps(extracted_data, indent=2))
    print("\nStoring extracted data...")
    if store_invoice_data(extracted_data):
        print("Data stored successfully")
    else:
        print("Failed to store data")


if __name__ == "__main__":
    main()
