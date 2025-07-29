# main.py
import os
import re
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import requests # For making HTTP requests to GitHub API

# --- Configuration and Initialization ---

# Configure logging
logging.basicConfig(
    level=logging.INFO, # Set to logging.DEBUG for more verbose output
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize FastAPI app
app = FastAPI(
    title="GitHub-Aware AI Assistant (No External LLM)",
    description="A FastAPI-based assistant that filters commands and interacts with GitHub API directly.",
    version="1.0.0"
)

# GitHub API configuration
GITHUB_API_BASE_URL = "https://api.github.com"
# Personal Access Token for GitHub. Read from environment variable.
# Recommended for higher rate limits and accessing private repos (if needed).
# Ensure GITHUB_TOKEN is set in your Render environment variables.
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_HEADERS = {
    "Accept": "application/vnd.github.v3+json",
    "Authorization": f"token {GITHUB_TOKEN}" if GITHUB_TOKEN else ""
}

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
                    if command: # Add non-empty, stripped, and lowercased lines
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

    # Normalize query for better matching (e.g., remove punctuation)
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

# --- GitHub Interaction Functions ---

def search_github_repositories(query: str, per_page: int = 3) -> list:
    """Searches GitHub repositories based on a query."""
    params = {"q": query, "per_page": per_page, "sort": "stars", "order": "desc"}
    try:
        response = requests.get(f"{GITHUB_API_BASE_URL}/search/repositories", headers=GITHUB_HEADERS, params=params)
        response.raise_for_status() # Raise an exception for HTTP errors
        return response.json().get("items", [])
    except requests.exceptions.RequestException as e:
        logging.error(f"GitHub API error searching repositories for '{query}': {e}")
        return []

def get_repo_contents(owner: str, repo: str, path: str = ""):
    """Gets the contents of a repository path (file or directory)."""
    try:
        url = f"{GITHUB_API_BASE_URL}/repos/{owner}/{repo}/contents/{path}"
        response = requests.get(url, headers=GITHUB_HEADERS)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"GitHub API error getting contents for {owner}/{repo}/{path}: {e}")
        return None

def fetch_file_content(download_url: str) -> str | None:
    """Fetches the raw content of a file from its download_url."""
    try:
        response = requests.get(download_url, headers=GITHUB_HEADERS)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        logging.error(f"GitHub API error fetching file content from {download_url}: {e}")
        return None

def process_github_query(query: str) -> str:
    """
    Processes the query to interact with GitHub API.
    This is where the "intelligence" of understanding the query and
    deciding what to fetch from GitHub lies.
    Currently, it's rule-based and looks for keywords.
    """
    response_parts = []
    query_lower = query.lower()

    # --- Rule 1: Search for general repositories ---
    if "repository" in query_lower or "repo" in query_lower or "project" in query_lower:
        search_term = next((word for word in query_lower.split() if word not in ["repository", "repo", "project", "github", "search", "for"]), query_lower)
        response_parts.append(f"Searching GitHub for repositories related to '{search_term}'...")
        repos = search_github_repositories(search_term)
        if repos:
            response_parts.append("Found these top repositories:")
            for i, repo in enumerate(repos[:5]): # Limit to top 5 for brevity
                response_parts.append(f"- {repo['full_name']}: {repo['html_url']} (Stars: {repo['stargazers_count']}) - {repo['description'] or 'No description'}")
        else:
            response_parts.append("No repositories found for your query.")
        return "\n".join(response_parts)

    # --- Rule 2: Fetch specific file content (e.g., "README.md") if repo is specified ---
    # This rule is simple: expects "get content of <file> from <owner>/<repo>"
    match_file_content = re.search(r"get content of (\S+) from (\S+)/(\S+)", query_lower)
    if match_file_content:
        file_name = match_file_content.group(1)
        owner = match_file_content.group(2)
        repo_name = match_file_content.group(3)

        response_parts.append(f"Attempting to fetch content of '{file_name}' from {owner}/{repo_name}...")
        contents = get_repo_contents(owner, repo_name, file_name)
        if contents and isinstance(contents, dict) and contents.get("type") == "file":
            download_url = contents.get("download_url")
            if download_url:
                file_content = fetch_file_content(download_url)
                if file_content:
                    # Limit file content display for very large files
                    if len(file_content) > 1000:
                        response_parts.append(f"\n--- Content of {file_name} (first 1000 chars) ---\n")
                        response_parts.append(file_content[:1000] + "\n...\n")
                    else:
                        response_parts.append(f"\n--- Content of {file_name} ---\n")
                        response_parts.append(file_content)
                    response_parts.append(f"\nFull file URL: {contents.get('html_url')}")
                else:
                    response_parts.append(f"Could not retrieve content for {file_name}.")
            else:
                response_parts.append(f"No direct download URL found for {file_name}.")
        else:
            response_parts.append(f"File '{file_name}' not found or is not a file in {owner}/{repo_name}.")
        return "\n".join(response_parts)

    # --- Rule 3: List files/folders in a specific repository ---
    match_list_contents = re.search(r"list (files|contents) in (\S+)/(\S+)(?: at path (.*))?", query_lower)
    if match_list_contents:
        owner = match_list_contents.group(2)
        repo_name = match_list_contents.group(3)
        path = match_list_contents.group(4) if match_list_contents.group(4) else ""

        response_parts.append(f"Listing contents of {owner}/{repo_name}/{path or '(root)'}...")
        contents = get_repo_contents(owner, repo_name, path)
        if contents and isinstance(contents, list):
            for item in contents:
                response_parts.append(f"- {item['type'].capitalize()}: {item['name']} ({item['html_url']})")
        else:
            response_parts.append(f"Could not list contents for {owner}/{repo_name}/{path}. It might not exist or be empty.")
        return "\n".join(response_parts)

    # --- Default Response if no specific GitHub action is matched ---
    return (
        f"Command allowed: '{query}'. However, this agent is currently configured to interact primarily with GitHub. "
        "Please specify if you'd like to search for repositories, list files, or get content from a specific repo/file. "
        "Example queries: "
        "\n- 'Find python web scraping repositories'"
        "\n- 'List contents in octocat/Spoon-Knife'"
        "\n- 'Get content of README.md from octocat/Spoon-Knife'"
    )

# --- FastAPI Endpoint ---

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Handles incoming chat requests:
    1. Loads allowed commands from 'allowed_commands.txt'.
    2. Checks if the user's message is allowed based on the loaded commands.
    3. If allowed, processes the request using GitHub API interaction logic.
    4. Returns the agent's response or a "Command not allowed" message.
    """
    user_message = request.message

    # Step 1: Check the query against allowed_commands.txt
    allowed_commands = load_allowed_commands(ALLOWED_COMMANDS_FILE)

    if not is_query_allowed(user_message, allowed_commands):
        logging.info(f"Blocked request: '{user_message}' - Command not allowed.")
        return {"response": "Command not allowed."}

    # Step 2: If allowed, process with GitHub interaction
    if not GITHUB_TOKEN:
        logging.error("GITHUB_TOKEN environment variable is not set. Cannot interact with GitHub API.")
        raise HTTPException(status_code=500, detail="GitHub interaction not configured. Please set GITHUB_TOKEN.")

    try:
        agent_response = process_github_query(user_message)
        logging.info(f"Successfully processed allowed request for: '{user_message}'")
        return {"response": agent_response}

    except Exception as e:
        logging.error(f"An unexpected error occurred while processing GitHub query for '{user_message}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred while processing your request. Please try again later or refine your query.")

# --- Local Development Server Runner (Optional) ---
if __name__ == "__main__":
    # To run locally, ensure you have set your GITHUB_TOKEN environment variable.
    # Example (in bash/zsh): export GITHUB_TOKEN="your_github_personal_access_token_here"
    logging.info("Starting FastAPI application for local development on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)