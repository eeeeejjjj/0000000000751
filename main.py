# main.py
import os
import re
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import uvicorn

# --- Configuration and Initialization ---

# Configure logging for better visibility during runtime and debugging
logging.basicConfig(
    level=logging.INFO, # Set to logging.DEBUG for more verbose output on query filtering
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize FastAPI app
app = FastAPI(
    title="AI Assistant",
    description="A FastAPI-based AI assistant with command filtering.",
    version="1.0.0"
)

# Initialize OpenAI client. It automatically reads OPENAI_API_KEY from environment variables.
# Ensure OPENAI_API_KEY is set in your Render environment variables.
openai_client = None
try:
    openai_client = OpenAI()
    logging.info("OpenAI client initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize OpenAI client: {e}. Please ensure OPENAI_API_KEY is set in your environment.")
    # The application will still start, but AI requests will fail if client is None.

# File path for allowed commands list
ALLOWED_COMMANDS_FILE = "allowed_commands.txt"

# --- Pydantic Models ---

# Defines the structure for incoming POST requests
class ChatRequest(BaseModel):
    message: str

# --- Helper Functions ---

def load_allowed_commands(file_path: str) -> set[str]:
    """
    Loads allowed commands from a plain text file.
    This function is called for each request to ensure immediate reflection of changes
    in the allowed_commands.txt file without requiring a server restart.
    """
    allowed_commands = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                command = line.strip().lower()
                if command: # Add non-empty, stripped, and lowercased lines
                    allowed_commands.add(command)
        logging.info(f"Loaded {len(allowed_commands)} allowed commands from {file_path} for current request.")
    except FileNotFoundError:
        logging.error(f"Allowed commands file '{file_path}' not found. No commands will be allowed.")
    except Exception as e:
        logging.error(f"Error loading allowed commands from '{file_path}': {e}. No commands will be allowed.")
    return allowed_commands

def is_query_allowed(query: str, allowed_commands: set[str]) -> bool:
    """
    Checks if the user's query contains at least one significant keyword or phrase
    from the `allowed_commands` list. Matching is case-insensitive.

    A query is considered allowed if:
    1. It contains any multi-word phrase present in `allowed_commands.txt`.
    2. Any individual word (tokenized) from the query is present in `allowed_commands.txt`.
    3. The entire query (lowercased) exactly matches a command in `allowed_commands.txt`.
    """
    if not allowed_commands:
        logging.warning("No allowed commands loaded; therefore, no queries are permitted.")
        return False # If no commands are loaded, nothing is allowed.

    query_lower = query.lower()

    # Strategy 1: Check for exact matches of multi-word commands (e.g., "web scraping")
    for command in allowed_commands:
        if ' ' in command and command in query_lower:
            logging.debug(f"Query allowed by multi-word command match: '{command}' found in '{query}'")
            return True

    # Strategy 2: Tokenize the query and check if any individual word is an allowed command.
    # Using regex for robust word tokenization (alphanumeric words only).
    query_words = set(re.findall(r'\b\w+\b', query_lower))

    for word in query_words:
        if word in allowed_commands:
            logging.debug(f"Query allowed by single-word command match: '{word}' found in '{query}'")
            return True

    # Strategy 3: Check if the entire query itself is an allowed command (e.g., "python")
    if query_lower in allowed_commands:
        logging.debug(f"Query allowed by exact query match: '{query_lower}'")
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
    3. If allowed, sends the message to OpenAI's GPT API.
    4. Returns the AI's response or a "Command not allowed" message.
    """
    user_message = request.message

    # Step 4: Check the query against allowed_commands.txt
    allowed_commands = load_allowed_commands(ALLOWED_COMMANDS_FILE)

    if not is_query_allowed(user_message, allowed_commands):
        logging.info(f"Blocked request: '{user_message}' - Command not allowed.")
        return {"response": "Command not allowed."}

    # Step 5: If allowed, process with OpenAI's GPT API
    if openai_client is None:
        logging.error("OpenAI client is not initialized. Cannot process AI request.")
        raise HTTPException(status_code=500, detail="AI service is currently unavailable. Please check server configuration.")

    try:
        # Define the system prompt to guide the AI's behavior and style
        system_prompt = (
            "You are a highly skilled and helpful AI assistant specialized in providing detailed, "
            "accurate information and comprehensive code samples. "
            "Your expertise covers programming, software development, data science, cybersecurity (ethical aspects), "
            "cloud computing, and general technology topics. "
            "When asked for code, provide clear, runnable examples in the requested language (Python, JavaScript, HTML, CSS, SQL, Shell, etc.). "
            "Always strive for clarity, accuracy, and thoroughness in your responses. "
            "If a user asks for something that subtly deviates but is still related to their allowed topic, "
            "try to guide them towards valuable, allowed information or code. "
            "Keep your responses professional and direct."
        )

        chat_completion = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",  # Consider "gpt-4o" or "gpt-4-turbo-preview" for higher quality if available and cost-effective.
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.7, # Controls creativity: lower for more focused, higher for more creative (0.7 is a good balance)
            max_tokens=1000, # Max length of the AI's response in tokens (adjust as needed)
        )

        ai_response = chat_completion.choices[0].message.content
        logging.info(f"Successfully processed allowed request for: '{user_message}'")
        return {"response": ai_response}

    except Exception as e:
        logging.error(f"Error processing OpenAI request for '{user_message}': {e}", exc_info=True)
        # Return a generic error to the user for security and user experience
        raise HTTPException(status_code=500, detail="An error occurred while processing your request with the AI backend. Please try again later.")

# --- Local Development Server Runner (Optional) ---
# This block is useful for testing the application locally.
# It will NOT be used when deploying to Render, as Render uses its own start command.
if __name__ == "__main__":
    # To run locally, ensure you have set your OPENAI_API_KEY environment variable.
    # Example (in bash/zsh): export OPENAI_API_KEY="your_actual_openai_api_key_here"
    logging.info("Starting FastAPI application for local development on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)