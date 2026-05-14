import json
from pathlib import Path

import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/recommend/stream"
USER_IDS_PATH = Path("data/user_ids.json")

st.set_page_config(page_title="Movie CRS", page_icon="🎬")

st.title("🎬 Movie Conversational Recommender")

mode = st.selectbox(
    "Recommendation mode",
    ["rag", "agent"],
)


def map_profile_id(profile_id: str) -> str | None:
    with open(USER_IDS_PATH, "r", encoding="utf-8") as file:
        user_ids = json.load(file)

    profile_map = {
        str(index): user_id
        for index, user_id in enumerate(user_ids)
    }

    return profile_map.get(profile_id)


if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Are you a returning user? (Yes/No)",
        }
    ]

if "stage" not in st.session_state:
    st.session_state.stage = "ask_user_type"

if "profile_id" not in st.session_state:
    st.session_state.profile_id = None

if "mapped_user_id" not in st.session_state:
    st.session_state.mapped_user_id = None

if "user_type" not in st.session_state:
    st.session_state.user_type = None


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


question = st.chat_input("Type your message...")

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    user_text = question.strip().lower()

    if st.session_state.stage == "ask_user_type":
        if (
            "return" in user_text
            or user_text == "yes"
            or user_text == "y"
        ):
            st.session_state.user_type = "returning"
            st.session_state.stage = "ask_profile_id"

            answer = (
                "Please enter your profile ID. "
                "Example: 0, 5, 12, or 42."
            )

        elif (
            "new" in user_text
            or user_text == "no"
            or user_text == "n"
        ):
            st.session_state.user_type = "new"
            st.session_state.stage = "ready"

            answer = "Great. What kind of movie would you like?"

        else:
            answer = (
                "Please answer with either:\n"
                "- Returning user / Yes\n"
                "- New user / No"
            )

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

        with st.chat_message("assistant"):
            st.markdown(answer)

    elif st.session_state.stage == "ask_profile_id":
        if question.strip().isdigit():
            profile_id = question.strip()
            mapped_user_id = map_profile_id(profile_id)

            if not mapped_user_id:
                answer = (
                    "Profile ID not found. "
                    "Would you like to continue as a new user?"
                )
            else:
                st.session_state.profile_id = profile_id
                st.session_state.mapped_user_id = mapped_user_id
                st.session_state.stage = "ready"

                answer = (
                    f"Great, I loaded your profile user {mapped_user_id}. "
                    "What kind of movie would you like?"
                )

        else:
            answer = "Please enter a numeric profile ID, for example: 12."

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

        with st.chat_message("assistant"):
            st.markdown(answer)

    else:
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""

            try:
                response = requests.post(
                    API_URL,
                    json={
                        "question": question,
                        "history": st.session_state.messages,
                        "user_id": st.session_state.profile_id,
                        "mode": mode,
                    },
                    stream=True,
                    timeout=120,
                )

                response.raise_for_status()

                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        text = chunk.decode("utf-8")
                        full_response += text
                        placeholder.markdown(full_response)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": full_response,
                    }
                )

            except requests.exceptions.ChunkedEncodingError:
                st.error(
                    "Backend stopped streaming early. "
                    "Check the FastAPI terminal."
                )

            except requests.exceptions.RequestException as e:
                st.error(f"Request failed: {e}")

            except Exception as e:
                st.error(f"Unexpected error: {e}")