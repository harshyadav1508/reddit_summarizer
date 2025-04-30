# --- START OF FILE app.py ---

import streamlit as st
from flask import Flask, request, jsonify
from threading import Thread
import requests
import praw
import os
from dotenv import load_dotenv
import google.generativeai as genai # Import Gemini library

st.set_page_config(layout="wide") # Use wider layout

# Load environment variables
load_dotenv()

# --- Flask Backend Setup ---
app = Flask(__name__)

# --- Reddit API Setup ---
try:
    reddit = praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent=os.getenv("REDDIT_USER_AGENT")
    )
    reddit.user.me() # Test authentication
    print("Reddit API Authentication Successful")
except Exception as e:
    print(f"Error initializing Reddit API: {e}")
    # Exit or handle gracefully if Reddit auth fails
    st.error(f"Failed to authenticate with Reddit API. Check credentials. Error: {e}")
    st.stop() # Stop streamlit execution if reddit fails


# --- Gemini API Setup ---
try:
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables.")
    genai.configure(api_key=gemini_api_key)
    gemini_model = genai.GenerativeModel('gemini-2.0-flash-lite') # Or use 'gemini-1.5-flash', etc.
    print("Gemini API Configured Successfully")
except Exception as e:
    print(f"Error initializing Gemini API: {e}")
    # Allow app to run, but summarization will fail
    gemini_model = None
    st.warning(f"Failed to configure Gemini API. Summarization will not work. Error: {e}")


# --- Comment Processing Functions (Unchanged) ---
def format_post_for_llm(post, query_keywords, max_top_comments=20, max_depth=3, min_score=3, min_length=20):
    # --- (Keep your existing format_post_for_llm function here) ---
    llm_input_string = f"Reddit Post URL: {post.url}\n" # Add URL for context
    llm_input_string += f"Post Title: {post.title}\n"
    llm_input_string += f"Post Body:\n{post.selftext}\n\n"
    llm_input_string += "--- Relevant Comments (threaded format) ---\n\n"

    # Ensure comments are loaded (handle potential exceptions)
    try:
        post.comments.replace_more(limit=0) # Remove 'MoreComments' at top level
        # Sort by score (or 'best'/'top' per Reddit's sorting)
        # PRAW default sort is often good enough, but explicit sort helps
        valid_comments = [c for c in post.comments if hasattr(c, 'score')]
        top_comments = sorted(valid_comments, key=lambda c: c.score, reverse=True)[:max_top_comments]
    except Exception as e:
        print(f"Error loading comments for post {post.id}: {e}")
        llm_input_string += "[Error fetching comments]\n"
        top_comments = [] # Ensure it's an empty list

    processed_comment_count = 0
    for comment in top_comments:
        thread_text = process_comment_recursive(
            comment, query_keywords, current_depth=0, max_depth=max_depth,
            min_score=min_score, min_length=min_length
        )
        if thread_text: # Only add if the thread wasn't completely filtered
            llm_input_string += thread_text + "---\n" # Separator between top-level threads
            processed_comment_count += 1

    if processed_comment_count == 0:
        llm_input_string += "[No relevant comments found based on filters]\n"

    return llm_input_string

def process_comment_recursive(comment, query_keywords, current_depth, max_depth, min_score, min_length):
    # --- (Keep your existing process_comment_recursive function here) ---
    # Base cases for recursion stop
    if current_depth > max_depth:
        return ""
    # Ensure it's a valid comment object before accessing attributes
    if not isinstance(comment, praw.models.Comment):
         return ""
    # Skip deleted comments ([deleted] body, [removed] author often)
    if comment.body is None or comment.body == '[deleted]' or comment.body == '[removed]':
        return ""
    # Skip if score is too low or body too short
    if comment.score < min_score or len(comment.body) < min_length:
        return ""

    # Optional: relevance check (uncomment if needed)
    # if query_keywords and not any(keyword.lower() in comment.body.lower() for keyword in query_keywords):
    #    return "" # Can be too aggressive, use with caution

    indent = "> " * current_depth
    # Clean comment body (handle potential NoneType, remove excessive newlines)
    cleaned_body = " ".join(comment.body.split()) if comment.body else ""
    formatted_comment = f"{indent}(Score: {comment.score}) {cleaned_body}\n"

    replies_text = ""
    # Process replies only if depth allows and replies exist
    if current_depth < max_depth and hasattr(comment, 'replies'):
        try:
            # Ensure replies are loaded, limit replacement to avoid deep calls here
            comment.replies.replace_more(limit=0) # Remove MoreComments within replies
            # Sort replies by score before processing
            # Filter out non-comment objects just in case
            valid_replies = [r for r in comment.replies if isinstance(r, praw.models.Comment)]
            sorted_replies = sorted(valid_replies, key=lambda r: r.score, reverse=True)

            for reply in sorted_replies:
                 # Recursive call needs to handle potential None return
                reply_text = process_comment_recursive(
                    reply, query_keywords, current_depth + 1, max_depth,
                    min_score, min_length
                )
                if reply_text: # Append only if valid text was returned
                     replies_text += reply_text

        except Exception as e:
            # Log error but continue processing other parts
            print(f"Error processing replies for comment {getattr(comment, 'id', 'N/A')}: {e}")
            # Optionally add an indicator in the text:
            # replies_text += f"{indent}> [Error loading replies]\n"

    return formatted_comment + replies_text

# --- Flask Route ---
@app.route('/summarize', methods=['POST'])
def summarize():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 415

    data = request.json
    reddit_url = data.get('url')
    # Optional: Allow passing keywords from frontend if needed later
    query_keywords = data.get('keywords', [])

    if not reddit_url:
        return jsonify({"error": "No URL provided"}), 400

    try:
        # --- Step 1: Extract Reddit post data ---
        submission = reddit.submission(url=reddit_url)
        # Prefetch comments to potentially speed up access later
        submission.comment_sort = 'top' # or 'best'
        submission.comments.replace_more(limit=None) # Load all comments initially if feasible, might be slow! Or keep limit=0 in recursive funcs.

        # --- Step 2: Format Comments for LLM Input ---
        # Pass actual keywords if you implement keyword filtering
        formatted_comments_for_llm = format_post_for_llm(
            submission,
            query_keywords=query_keywords, # Pass keywords here
            max_top_comments=20, # How many top-level threads to start with
            max_depth=3,
            min_score=3,
            min_length=25 # Slightly increased min length
        )

        # --- Step 3: Prepare Prompt for Gemini ---
        # Improved prompt: Be more specific about the desired output
        prompt = (
            "You are an AI assistant specialized in summarizing Reddit discussions.\n"
            "Below is the content of a Reddit post (title, body) and a structured representation of its comment threads.\n"
            "Comments are threaded using '>' for replies, and scores `(Score: X)` indicate community rating.\n\n"
            "--- Reddit Content Start ---\n"
            f"{formatted_comments_for_llm}"
            "--- Reddit Content End ---\n\n"
            "**Task:** Please provide a concise summary of the main points, opinions, arguments, and key takeaways from the entire discussion (post and comments).\n"
            "Focus on the most relevant and highly-rated information. If applicable, mention:\n"
            "- Overall sentiment or consensus.\n"
            "- Key pros and cons discussed.\n"
            "- Specific recommendations or advice given.\n"
            "- Any significant disagreements or alternative viewpoints.\n\n"
            "**Output Format:** Start with a brief overall summary paragraph, then use bullet points for key details if helpful. Be objective and base the summary *only* on the provided text."
        )

        # --- Step 4: Send prompt to Gemini API ---
        gemini_summary = "[Gemini Summarization Failed or Disabled]" # Default message
        if gemini_model: # Check if Gemini was initialized successfully
            try:
                print("\n--- Sending Prompt to Gemini ---")
                # print(prompt) # Uncomment to debug the full prompt
                print("--- End of Prompt ---")

                # Configure safety settings if needed (optional)
                # safety_settings = [
                #     {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                #     {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                # ]
                # response = gemini_model.generate_content(prompt, safety_settings=safety_settings)

                response = gemini_model.generate_content(prompt)

                # Check for safety blocks or empty response
                if not response.parts:
                     gemini_summary = "[Gemini response blocked due to safety filters or returned empty]"
                     print(f"Gemini Warning: {response.prompt_feedback}")
                else:
                    gemini_summary = response.text
                    print("Gemini Summarization Successful")

            except Exception as e:
                print(f"Error calling Gemini API: {e}")
                gemini_summary = f"[Error during Gemini API call: {e}]"
        else:
            print("Gemini API not configured. Skipping summarization.")


        # --- Step 5: Return Results ---
        return jsonify({
            "title": submission.title,
            "url": submission.url, # Include URL in response
            "body": submission.selftext,
            "formatted_comments_display": formatted_comments_for_llm, # Use the version including post info for display
            "gemini_summary": gemini_summary
        })

    except praw.exceptions.InvalidURL:
         return jsonify({"error": "Invalid Reddit URL provided."}), 400
    except praw.exceptions.RedditAPIException as e:
        print(f"Reddit API Error: {e}")
        return jsonify({"error": f"Reddit API error: {e}"}), 500
    except requests.exceptions.ConnectionError as e:
        print(f"Connection Error: {e}")
        return jsonify({"error": "Network error connecting to Reddit or Gemini."}), 503
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback to Flask console
        return jsonify({"error": f"An unexpected server error occurred: {e}"}), 500

# --- Function to Run Flask ---
def run_flask():
    # Use '0.0.0.0' to make it accessible on the network if needed,
    # otherwise '127.0.0.1' is safer for local development.
    app.run(host='127.0.0.1', port=5000) # Running on default Flask dev server

# --- Start Flask in a Separate Thread ---
# Ensure Flask starts only once, typically when the script runs
if 'flask_thread' not in st.session_state:
    st.session_state.flask_thread = Thread(target=run_flask, daemon=True)
    st.session_state.flask_thread.start()
    print("Flask thread started.")

# ======== Streamlit Frontend ========



st.title("Reddit Discussion Summarizer (using Gemini)")
st.write("Enter a Reddit post link to get a summary of the discussion.")

# --- Input Area ---
col1, col2 = st.columns([3, 1]) # Make input column wider
with col1:
    reddit_url = st.text_input("Reddit Post Link:", key="reddit_url_input")
with col2:
    st.write("") # Spacer
    st.write("") # Spacer
    summarize_button = st.button("Summarize Discussion", key="summarize_btn")

# --- Display Area ---
if summarize_button and reddit_url:
    # Basic URL validation (very simple)
    if "reddit.com" not in reddit_url or "/comments/" not in reddit_url:
        st.error("Please enter a valid Reddit post link (e.g., https://www.reddit.com/r/subreddit/comments/...).")
    else:
        try:
            backend_url = "http://127.0.0.1:5000/summarize"
            # Use a spinner while waiting for the backend
            with st.spinner(f"Fetching Reddit data and summarizing with Gemini... Please wait."):
                response = requests.post(backend_url, json={"url": reddit_url}, timeout=120) # Increase timeout

            if response.status_code == 200:
                data = response.json()

                # --- Main Content Area (Title, Body, Summary) ---
                st.subheader(f"Post Title: {data.get('title', 'N/A')}")
                st.markdown(f"[Link to Post]({data.get('url', '#')})") # Add link

                st.subheader("Original Post Body")
                st.markdown(data.get('body', '*No body text*')) # Use markdown for better formatting

                st.divider() # Visual separator

                st.subheader("✨ Gemini Summary ✨")
                st.markdown(data.get('gemini_summary', '*No summary generated.*'))

                # --- Sidebar (Formatted Comments) ---
                with st.sidebar:
                    st.subheader("Processed Comments Preview")
                    st.write("This shows the filtered and formatted comment threads used for summarization.")
                    # Use text_area for scrollable view
                    st.text_area(
                        "Comment Threads",
                        data.get('formatted_comments_display', '*No comments processed or found.*'),
                        height=600, # Adjust height as needed
                        key="comments_display_area"
                    )

            else:
                try:
                    error_data = response.json()
                    st.error(f"Error from backend (Status {response.status_code}): {error_data.get('error', 'Unknown error')}")
                except requests.exceptions.JSONDecodeError:
                    st.error(f"Error: Received non-JSON response from backend (Status {response.status_code}). Check Flask logs.")
                except Exception as e:
                     st.error(f"Error processing backend response: {e}")

        except requests.exceptions.Timeout:
            st.error("Error: The request to the backend timed out. The post might be too large or the server is busy.")
        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to the backend API at {backend_url}. Is Flask running? Error: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred in the Streamlit frontend: {e}")
            import traceback
            st.error(traceback.format_exc()) # Show detailed error in UI for debugging

elif summarize_button and not reddit_url:
    st.warning("Please enter a Reddit link first.")


# Add some footer or info text
st.sidebar.markdown("---")
st.sidebar.info("This app uses PRAW to fetch Reddit data and Google Gemini for summarization.")

# --- END OF FILE app.py ---