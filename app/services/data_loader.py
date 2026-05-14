import json
from pathlib import Path


def load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            rows.append(json.loads(line))
    return rows


def load_text(path: str | Path) -> str:
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def get_conversation_by_id(content: str, conversation_id: int) -> str:
    blocks = content.strip().split("\n\n")
    current_id = None
    conversation = []

    for block in blocks:
        if block.strip().isdigit():
            if current_id == conversation_id:
                return "\n\n".join(conversation)

            current_id = int(block.strip())
            conversation = []
        else:
            conversation.append(block)

    if current_id == conversation_id:
        return "\n\n".join(conversation)

    return ""