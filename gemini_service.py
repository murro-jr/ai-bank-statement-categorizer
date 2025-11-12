import json
import base64
from enum import Enum
from google import genai
from google.genai import types
from google.genai.errors import APIError

from dotenv import load_dotenv

load_dotenv()
# --- 1. Define Transaction Category Enum (Equivalent to TransactionCategory in TS) ---
# In Python, we define the possible categories using an Enum.
class TransactionCategory(str, Enum):
    """
    Defines the allowed categories for transactions.
    This replaces the imported 'TransactionCategory' from the original code.
    """
    IncomeRevenue = "Income / Revenue"
    OperatingExpenses = "Operating Expenses"
    COGS = "Cost of Goods Sold (COGS)"
    FinancialBanking = "Financial and Banking"
    AssetsCapital = "Assets and Capital"
    Liabilities = "Liabilities"
    TaxesGovernment = "Taxes and Government Payments"
    Other = "Other / Miscellaneous"

# --- 2. Initialize the Client and Define the Schema ---

# Initializes the Gemini Client. 
# It automatically looks for the GEMINI_API_KEY environment variable.
try:
    client = genai.Client()
except Exception as e:
    print(f"Error initializing client: {e}")
    print("Please ensure the GEMINI_API_KEY environment variable is set.")
    exit()

# Defines the JSON Schema for the structured output.
# This replaces the usage of `Type.ARRAY`, `Type.OBJECT`, etc.
response_schema = types.Schema(
    type=types.Type.ARRAY,
    items=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "date": types.Schema(
                type=types.Type.STRING,
                description="Transaction date in YYYY-MM-DD format.",
            ),
            "description": types.Schema(
                type=types.Type.STRING,
                description="A brief description of the transaction.",
            ),
            "amount": types.Schema(
                type=types.Type.NUMBER,
                description="The transaction amount. Use negative numbers for debits/expenses and positive numbers for credits/income.",
            ),
            "category": types.Schema(
                type=types.Type.STRING,
                # Converts the Python Enum values to a list of strings for the schema enum
                enum=[category.value for category in TransactionCategory], 
                description="The category of the transaction.",
            ),
        },
        required=['date', 'description', 'amount', 'category'],
    ),
)

# --- 3. Define the Main Extraction Function ---

def extract_and_categorize_transactions(
    file_path: str,
    mime_type: str
) -> list[dict]:
    """
    Extracts, classifies, and categorizes bank transactions from a file 
    using the Gemini API with structured JSON output.

    Args:
        file_path: The local path to the bank statement (e.g., 'statement.pdf').
        mime_type: The MIME type of the file (e.g., 'application/pdf').

    Returns:
        A list of transaction dictionaries extracted from the document.
    """
    print(f"Loading file: {file_path}")
    
    # Read the file and encode it to base64 for inline data transfer
    try:
        with open(file_path, "rb") as f:
            file_bytes = f.read()
            base64_data = base64.b64encode(file_bytes).decode("utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at path: {file_path}")

    # Prepare the parts for the multimodal prompt
    file_part = types.Part.from_bytes(data=file_bytes, mime_type=mime_type)

    text_string = f"""You are an expert financial analyst. Analyze the provided bank statement file.
        Extract every transaction, including its date, description, and amount.
        For each transaction, determine if it is a credit (income) or a debit (expense).
        Finally, categorize each transaction into one of the specified categories: 
        {', '.join([c.value for c in TransactionCategory])}.
        Provide the output as a valid JSON array of objects, conforming strictly to the provided schema.
        Ensure that amounts for expenses are represented as negative numbers, and income as positive numbers.
        """
    text_part = types.Part.from_text(text=text_string)

    # Configure the generation settings
    config = types.GenerateContentConfig(
        response_mime_type='application/json',
        response_schema=response_schema,
    )

    print("Sending request to Gemini API...")
    try:
        # Call the API
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[file_part, text_part],
            config=config
        )
    except APIError as e:
        print(f"Gemini API Error: {e}")
        return []
    
    # Process the structured JSON output
    try:
        json_string = response.text
        # The response text should be a valid JSON string due to the config
        parsed_json = json.loads(json_string) 
        
        # print("\n--- Extracted Transactions (JSON) ---")
        # print(json.dumps(parsed_json, indent=2))
        
        return parsed_json
    except json.JSONDecodeError as e:
        print("Failed to parse Gemini response as JSON.")
        print(f"Error: {e}")
        print("Raw response text:")
        print(response.text)
        raise ValueError("The AI model returned an invalid JSON format.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return []