# main.py
import os
import re
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import google.generativeai as genai # New import for Google Gemini

# --- Configuration and Initialization ---

# Configure logging
logging.basicConfig(
    level=logging.INFO, # Set to logging.DEBUG for more verbose output on query filtering
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize FastAPI app
app = FastAPI(
    title="Gemini-Powered AI Assistant with Filtering",
    description="A FastAPI-based AI assistant filtering commands and using Google Gemini for responses.",
    version="1.0.0"
)

# Initialize Google Gemini client.
# It automatically reads GOOGLE_API_KEY from environment variables.
# Ensure GOOGLE_API_KEY is set in your Render environment variables.
try:
    # Attempt to configure Gemini client immediately at startup
    # This will fail if GOOGLE_API_KEY is not set, but the app will still start.
    # Requests to /chat will then raise an HTTPException.
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logging.error("GOOGLE_API_KEY environment variable is not set. Gemini API will not function.")
    else:
        genai.configure(api_key=api_key)
        logging.info("Google Gemini client configured successfully.")
except Exception as e:
    logging.error(f"Failed to configure Google Gemini client at startup: {e}. "
                  "Please ensure GOOGLE_API_KEY is set in your Render environment variables.")

# Try to load the model (optional, but good for early error detection)
gemini_model = None
if api_key:
    try:
        gemini_model = genai.GenerativeModel('gemini-pro') # Or 'gemini-1.5-pro' for higher capabilities
        logging.info("Gemini model 'gemini-pro' loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load Gemini model 'gemini-pro': {e}. Check GOOGLE_API_KEY and model availability.")


# File path for allowed commands list
ALLOWED_COMMANDS_FILE = "allowed_commands.txt"

# --- Pydantic Models ---

# Defines the structure for incoming POST requests
class ChatRequest(BaseModel):
    message: str

# --- Helper Functions (Command Filtering) ---

def load_allowed_commands(file_path: str) -> set[str]:
    """
    Loads allowed commands from a plain text file.
    This function is called for each request to ensure immediate reflection of changes
    in the allowed_commands.txt file without requiring a server restart.
    """
    allowed_commands = set()
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    command = line.strip().lower()
                    if command:
                        allowed_commands.add(command)
            logging.info(f"Loaded {len(allowed_commands)} allowed commands from {file_path} for current request.")
        else:
            logging.warning(f"Allowed commands file '{file_path}' not found. No commands will be allowed initially.")
    except Exception as e:
        logging.error(f"Error loading allowed commands from '{file_path}': {e}. No commands will be allowed.")
    return allowed_commands

def is_query_allowed(query: str, allowed_commands: set[str]) -> bool:
    """
    Checks if the user's query contains at least one significant keyword or phrase
    from the `allowed_commands` list. Matching is case-insensitive.
    """
    if not allowed_commands:
        logging.warning("No allowed commands loaded; therefore, no queries are permitted.")
        return False

    query_lower = query.lower()
    normalized_query = re.sub(r'[^a-z0-9\s]', '', query_lower)
    normalized_query_words = set(normalized_query.split())

    # Check for multi-word phrases first
    for command in allowed_commands:
        if ' ' in command:
            if command in query_lower:
                logging.debug(f"Query allowed by multi-word command match: '{command}' found in '{query}'")
                return True

    # Check for individual words or single-word commands
    for word in normalized_query_words:
        if word in allowed_commands:
            logging.debug(f"Query allowed by single-word command match: '{word}' found in '{query}'")
            return True

    # Check if the entire (normalized) query itself is an allowed command
    if normalized_query in allowed_commands:
        logging.debug(f"Query allowed by exact normalized query match: '{normalized_query}'")
        return True

    logging.info(f"Query '{query}' did not contain any allowed commands or keywords.")
    return False

# --- FastAPI Endpoint ---

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Handles incoming chat requests:
    1. Loads allowed commands from 'allowed_commands.txt'.
    2. Checks if the user's message is allowed based on the loaded commands.
    3. If allowed, forwards the message to Google Gemini for response generation.
    4. Returns the AI's response or a "Command not allowed" message.
    """
    user_message = request.message

    # Step 1: Check the query against allowed_commands.txt
    allowed_commands = load_allowed_commands(ALLOWED_COMMANDS_FILE)

    if not is_query_allowed(user_message, allowed_commands):
        logging.info(f"Blocked request: '{user_message}' - Command not allowed.")
        return {"response": "Command not allowed."}

    # Step 2: If allowed, process with Google Gemini API
    if gemini_model is None:
        logging.error("Google Gemini model is not initialized. Cannot process AI request.")
        raise HTTPException(status_code=500, detail="AI service is currently unavailable. Please ensure GOOGLE_API_KEY is set and valid.")

    try:
        # Define the system prompt for Gemini
        system_prompt = (
            "You are a highly capable AI assistant that provides detailed, accurate information and code samples. "
            "Your knowledge base is vast, allowing you to answer questions on programming, software engineering, "
            "data science, cloud computing, and more. You can 'search' for information and 'organize' code samples "
            "to provide clean, well-structured responses. If asked for code, provide runnable examples in the requested language. "
            "Your responses should be comprehensive and helpful, as if you have learned from all available resources "
            "(including public code repositories and documentation). Do not express limitations on your knowledge; "
            "instead, aim to provide the best possible answer based on your capabilities."
        )
        
        # Combine system prompt and user message for Gemini
        # Gemini often prefers the prompt to be part of the user's turn
        # or as a single combined string, depending on the client library's
        # model.generate_content implementation for system instructions.
        # For simple text generation, a combined prompt works well.
        full_prompt = f"{system_prompt}\n\nUser Query: {user_message}"

        # Make the request to Gemini API
        response = gemini_model.generate_content(
            full_prompt,
            safety_settings={
                "HARASSMENT": "BLOCK_NONE",
                "HATE": "BLOCK_NONE",
                "SEXUAL": "BLOCK_NONE",
                "DANGEROUS": "BLOCK_NONE",
            },
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,  # Adjust for creativity (0.0 for deterministic, 1.0 for highly creative)
                max_output_tokens=2000 # Max length of the AI's response
            )
        )

        ai_response = response.text # Get the generated text content
        logging.info(f"Successfully processed allowed request for: '{user_message}'")
        return {"response": ai_response}

    except Exception as e:
        logging.error(f"Error processing Google Gemini request for '{user_message}': {e}", exc_info=True)
        # Return a generic error to the user for security and user experience
        raise HTTPException(status_code=500, detail="An error occurred while processing your request with the AI backend. Please try again later.")

# --- Local Development Server Runner (Optional) ---
if __name__ == "__main__":
    # To run locally, ensure you have set your GOOGLE_API_KEY environment variable.
    # Example (in bash/zsh): export GOOGLE_API_KEY="your_actual_gemini_api_key_here"
    logging.info("Starting FastAPI application for local development on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)