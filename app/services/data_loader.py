import json
from pathlib import Path


def load_json(path: str | Path) -> dict:
    """
    Loads a standard JSON file and returns it as a dictionary.
    """

    # Open the JSON file using UTF-8 encoding
    with open(path, "r", encoding="utf-8") as file:

        # Parse JSON content into Python dictionary
        return json.load(file)


def load_jsonl(path: str | Path) -> list[dict]:
    """
    Loads a JSONL (JSON Lines) file.

    Each line in the file must contain
    a valid JSON object.
    """

    rows = []

    # Open the JSONL file
    with open(path, "r", encoding="utf-8") as file:

        # Read file line by line
        for line in file:

            # Convert JSON string into Python dictionary
            rows.append(json.loads(line))

    return rows


def load_text(path: str | Path) -> str:
    """
    Reads and returns the full content
    of a text file.
    """

    # Open text file
    with open(path, "r", encoding="utf-8") as file:

        # Return entire file content as string
        return file.read()


def get_conversation_by_id(content: str, conversation_id: int) -> str:
    """
    Extracts a conversation block from a text dataset
    using the conversation ID.

    Expected format:
    - Conversation IDs are numeric lines
    - Messages follow until the next numeric ID
    """

    # Split conversations using double line breaks
    blocks = content.strip().split("\n\n")

    current_id = None
    conversation = []

    # Iterate through all blocks
    for block in blocks:

        # Detect conversation ID blocks
        if block.strip().isdigit():

            # If target conversation was already found,
            # return the collected conversation
            if current_id == conversation_id:
                return "\n\n".join(conversation)

            # Update current conversation ID
            current_id = int(block.strip())

            # Reset conversation accumulator
            conversation = []

        else:
            # Append conversation text block
            conversation.append(block)

    # Handle case where target conversation
    # is the last one in the file
    if current_id == conversation_id:
        return "\n\n".join(conversation)

    # Return empty string if conversation is not found
    return ""