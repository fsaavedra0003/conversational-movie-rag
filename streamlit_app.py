import json
from pathlib import Path

# HTTP requests to communicate with FastAPI backend
import requests

# Streamlit UI framework
import streamlit as st


# FastAPI streaming recommendation endpoint
API_URL = "http://127.0.0.1:8000/recommend/stream"

# Path to stored user IDs dataset
USER_IDS_PATH = Path("data/user_ids.json")


# Configure Streamlit page
st.set_page_config(
    page_title="Movie CRS",
    page_icon="🎬",
)

# Main application title
st.title("🎬 Movie Conversational Recommender")


# Dropdown to select recommendation architecture
# - rag   -> standard RAG recommender
# - agent -> agentic recommender with tools
mode = st.selectbox(
    "Recommendation mode",
    ["rag", "agent"],
)


def map_profile_id(profile_id: str) -> str | None:
    """
    Maps numeric frontend profile IDs
    to real dataset user IDs.
    """

    # Load dataset user IDs
    with open(USER_IDS_PATH, "r", encoding="utf-8") as file:
        user_ids = json.load(file)

    # Create mapping:
    # "0" -> dataset_user_id_1
    # "1" -> dataset_user_id_2
    profile_map = {
        str(index): user_id
        for index, user_id in enumerate(user_ids)
    }

    # Return mapped user ID
    return profile_map.get(profile_id)


# Initialize chat history only once
if "messages" not in st.session_state:

    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Are you a returning user? (Yes/No)",
        }
    ]


# Conversation stage controller
#
# Stages:
# - ask_user_type
# - ask_profile_id
# - ready
if "stage" not in st.session_state:
    st.session_state.stage = "ask_user_type"


# Stores numeric frontend profile ID
if "profile_id" not in st.session_state:
    st.session_state.profile_id = None


# Stores mapped dataset user ID
if "mapped_user_id" not in st.session_state:
    st.session_state.mapped_user_id = None


# Stores whether user is:
# - new
# - returning
if "user_type" not in st.session_state:
    st.session_state.user_type = None


# Render previous chat messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Chat input field
question = st.chat_input("Type your message...")


# Execute logic only when user sends a message
if question:

    # Save user message into session history
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    # Display user message in UI
    with st.chat_message("user"):
        st.markdown(question)

    # Normalize text for easier matching
    user_text = question.strip().lower()

    # =========================
    # STAGE 1:
    # Ask if user is returning or new
    # =========================
    if st.session_state.stage == "ask_user_type":

        # Returning user flow
        if (
            "return" in user_text
            or user_text == "yes"
            or user_text == "y"
        ):

            st.session_state.user_type = "returning"

            # Move to next stage
            st.session_state.stage = "ask_profile_id"

            answer = (
                "Please enter your profile ID. "
                "Example: 0, 5, 12, or 42."
            )

        # New user flow
        elif (
            "new" in user_text
            or user_text == "no"
            or user_text == "n"
        ):

            st.session_state.user_type = "new"

            # Skip profile loading
            st.session_state.stage = "ready"

            answer = "Great. What kind of movie would you like?"

        # Invalid response
        else:

            answer = (
                "Please answer with either:\n"
                "- Returning user / Yes\n"
                "- New user / No"
            )

        # Save assistant response
        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

        # Render assistant response
        with st.chat_message("assistant"):
            st.markdown(answer)

    # =========================
    # STAGE 2:
    # Ask for profile ID
    # =========================
    elif st.session_state.stage == "ask_profile_id":

        # Validate numeric input
        if question.strip().isdigit():

            profile_id = question.strip()

            # Convert frontend profile ID
            # into dataset user ID
            mapped_user_id = map_profile_id(profile_id)

            # Handle missing profile
            if not mapped_user_id:

                answer = (
                    "Profile ID not found. "
                    "Would you like to continue as a new user?"
                )

            else:
                # Save valid profile info
                st.session_state.profile_id = profile_id

                st.session_state.mapped_user_id = mapped_user_id

                # System ready for recommendations
                st.session_state.stage = "ready"

                answer = (
                    f"Great, I loaded your profile user "
                    f"{mapped_user_id}. "
                    "What kind of movie would you like?"
                )

        else:
            # Reject non-numeric profile IDs
            answer = (
                "Please enter a numeric profile ID, "
                "for example: 12."
            )

        # Save assistant response
        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

        # Render assistant response
        with st.chat_message("assistant"):
            st.markdown(answer)

    # =========================
    # STAGE 3:
    # Recommendation chat loop
    # =========================
    else:

        with st.chat_message("assistant"):

            # Placeholder used for streaming UI updates
            placeholder = st.empty()

            full_response = ""

            try:
                # Send request to FastAPI backend
                response = requests.post(
                    API_URL,
                    json={
                        "question": question,

                        # Full chat history
                        "history": st.session_state.messages,

                        # Frontend profile ID
                        "user_id": st.session_state.profile_id,

                        # rag or agent mode
                        "mode": mode,
                    },
                    stream=True,
                    timeout=120,
                )

                # Raise exception if request failed
                response.raise_for_status()

                # Stream chunks progressively
                for chunk in response.iter_content(
                    chunk_size=1024
                ):

                    if chunk:

                        # Decode streamed bytes
                        text = chunk.decode("utf-8")

                        # Accumulate response
                        full_response += text

                        # Update UI in real time
                        placeholder.markdown(full_response)

                # Save final assistant response
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": full_response,
                    }
                )

            # Handle streaming interruptions
            except requests.exceptions.ChunkedEncodingError:

                st.error(
                    "Backend stopped streaming early. "
                    "Check the FastAPI terminal."
                )

            # Handle request/network errors
            except requests.exceptions.RequestException as e:

                st.error(f"Request failed: {e}")

            # Catch unexpected runtime errors
            except Exception as e:

                st.error(f"Unexpected error: {e}")