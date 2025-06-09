# --- START OF FILE app.py ---

import streamlit as st
from flask import Flask, request, jsonify
from threading import Thread
import requests
import praw
import prawcore # Import for more specific PRAW exceptions
import os
from dotenv import load_dotenv
import google.generativeai as genai # Import Gemini library
import traceback # For detailed error logging

st.set_page_config(layout="wide") # Use wider layout

# Load environment variables
load_dotenv()

# --- Flask Backend Setup ---
app = Flask(__name__)

# --- Reddit API Setup ---
# Encapsulate initialization to avoid running it multiple times if script reruns weirdly
if 'reddit_client' not in st.session_state:
    try:
        st.session_state.reddit_client = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent=os.getenv("REDDIT_USER_AGENT")
        )
        st.session_state.reddit_client.user.me() # Test authentication
        print("Reddit API Authentication Successful")
        st.session_state.reddit_error = None
    except Exception as e:
        print(f"Error initializing Reddit API: {e}")
        st.session_state.reddit_client = None
        st.session_state.reddit_error = f"Failed to authenticate with Reddit API. Check credentials. Error: {e}"
        # Display error early in Streamlit if needed
        st.error(st.session_state.reddit_error)
        st.stop()

# --- Gemini API Setup ---
if 'gemini_model' not in st.session_state:
    try:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        genai.configure(api_key=gemini_api_key)
        # Use a specific model name
        st.session_state.gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest') # Use a known working model like 1.5 Flash
        print("Gemini API Configured Successfully")
        st.session_state.gemini_error = None
    except Exception as e:
        print(f"Error initializing Gemini API: {e}")
        st.session_state.gemini_model = None
        st.session_state.gemini_error = f"Failed to configure Gemini API. Summarization will not work. Error: {e}"
        # Display warning in Streamlit
        st.warning(st.session_state.gemini_error)

# Access clients from session state within functions/routes
reddit = st.session_state.get('reddit_client')
gemini_model = st.session_state.get('gemini_model')


# --- Comment Processing Functions (Unchanged) ---
def format_post_for_llm(post, query_keywords, max_top_comments=20, max_depth=3, min_score=3, min_length=20):
    # --- (Keep your existing format_post_for_llm function here) ---
    llm_input_string = f"Reddit Post URL: {post.url}\n" # Add URL for context
    llm_input_string += f"Post Title: {post.title}\n"
    llm_input_string += f"Post Body:\n{post.selftext}\n\n"
    llm_input_string += "--- Relevant Comments (threaded format) ---\n\n"

    # Ensure comments are loaded (handle potential exceptions)
    try:
        # Set a sort order before accessing comments
        post.comment_sort = 'top' # Explicitly sort by top
        post.comments.replace_more(limit=0) # Remove 'MoreComments' at top level
        # Fetch comments after setting sort
        comments_list = post.comments.list()
        # Filter and sort valid comments
        valid_comments = [c for c in comments_list if isinstance(c, praw.models.Comment) and hasattr(c, 'score') and c.body and c.body not in ('[deleted]', '[removed]')]
        top_comments = sorted(valid_comments, key=lambda c: c.score, reverse=True)[:max_top_comments]
    except prawcore.exceptions.NotFound:
         llm_input_string += "[Comments not found or post deleted]\n"
         top_comments = []
    except Exception as e:
        print(f"Error loading/sorting comments for post {post.id}: {e}")
        llm_input_string += f"[Error fetching comments: {e}]\n"
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

    if processed_comment_count == 0 and not top_comments:
        llm_input_string += "[No relevant comments found based on filters or comments unavailable]\n"
    elif processed_comment_count == 0:
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
    # Skip deleted/removed comments or those without a body/score
    if not hasattr(comment, 'body') or comment.body is None or comment.body in ('[deleted]', '[removed]') or not hasattr(comment, 'score'):
        return ""
    # Skip if score is too low or body too short
    if comment.score < min_score or len(comment.body) < min_length:
        return ""

    # Optional: relevance check (uncomment if needed)
    # if query_keywords and not any(keyword.lower() in comment.body.lower() for keyword in query_keywords):
    #    return "" # Can be too aggressive, use with caution

    indent = "> " * current_depth
    # Clean comment body (remove excessive newlines)
    cleaned_body = " ".join(comment.body.split())
    formatted_comment = f"{indent}(Score: {comment.score}) {cleaned_body}\n"

    replies_text = ""
    # Process replies only if depth allows and replies exist and are loaded
    if current_depth < max_depth and hasattr(comment, 'replies'):
        try:
            # Ensure replies are loaded, limit replacement to avoid deep calls here
            comment.replies.replace_more(limit=0) # Remove MoreComments within replies
            # Filter out non-comment objects just in case
            valid_replies = [r for r in comment.replies if isinstance(r, praw.models.Comment)]
            # Sort replies by score before processing
            sorted_replies = sorted(valid_replies, key=lambda r: r.score if hasattr(r,'score') else -1, reverse=True) # Handle potential missing score on replies

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

# --- URL Resolution Function ---
def resolve_reddit_url(url):
    """
    Attempts to resolve a potentially shortened or tracked Reddit URL
    to its canonical submission URL using HTTP redirects.
    Returns the resolved URL or None if resolution fails or is not a Reddit URL.
    """
    if not url or "reddit.com" not in url:
        return None # Not a reddit URL

    try:
        # Use HEAD request to be faster, follow redirects
        headers = {'User-Agent': os.getenv("REDDIT_USER_AGENT", 'URLResolver/1.0')} # Use a user agent
        response = requests.head(url, allow_redirects=True, timeout=10, headers=headers)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        final_url = response.url
        # Basic check if the resolved URL looks like a comment page
        if "reddit.com" in final_url and "/comments/" in final_url:
            print(f"Resolved URL: {url} -> {final_url}")
            return final_url
        else:
            print(f"URL {url} resolved to {final_url}, which doesn't look like a standard post URL.")
            return None # Resolved URL doesn't fit expected pattern
    except requests.exceptions.Timeout:
        print(f"Timeout while resolving URL: {url}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error resolving URL {url}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error resolving URL {url}: {e}")
        return None

# --- Flask Route ---
@app.route('/summarize', methods=['POST'])
def summarize():
    if not reddit: # Check if Reddit client is initialized
        return jsonify({"error": "Reddit client not available. Check backend logs."}), 503
    if not gemini_model: # Check if Gemini model is initialized
         st.warning("Gemini model not available. Proceeding without summarization.") # Or return error

    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 415

    data = request.json
    original_url = data.get('url')
    query_keywords = data.get('keywords', []) # Keep this if needed later

    if not original_url:
        return jsonify({"error": "No URL provided"}), 400

    # --- Step 0: Resolve URL ---
    resolved_url = resolve_reddit_url(original_url)
    if not resolved_url:
        return jsonify({"error": f"Could not resolve the provided URL or it doesn't lead to a valid Reddit post: {original_url}"}), 400

    try:
        # --- Step 1: Extract Reddit post data ---
        print(f"Fetching submission for resolved URL: {resolved_url}")
        submission = reddit.submission(url=resolved_url)
        # Access an attribute to force fetching and check validity
        _ = submission.title

        # --- Step 2: Format Comments for LLM Input ---
        formatted_comments_for_llm = format_post_for_llm(
            submission,
            query_keywords=query_keywords, # Pass keywords here
            max_top_comments=25, # Slightly more top comments
            max_depth=3,
            min_score=2, # Lower min score slightly
            min_length=20 # Keep min length reasonable
        )

        # --- Step 3: Prepare Prompt for Gemini ---
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
            "**Output Format:** Start with a brief overall summary paragraph, then use bullet points for key details if helpful. Be objective and base the summary *only* on the provided text and try to give example from post for the bullet points."
        )

        # --- Step 4: Send prompt to Gemini API ---
        gemini_summary = "[Gemini Summarization Failed or Disabled]" # Default message
        if gemini_model: # Check if Gemini was initialized successfully
            try:
                print("\n--- Sending Prompt to Gemini ---")
                # print(prompt) # Uncomment to debug the full prompt
                print(f"Prompt length (approx chars): {len(prompt)}")
                print("--- End of Prompt ---")

                # Configure safety settings - block fewer categories if needed, adjust thresholds
                # safety_settings = [
                #     {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
                #     {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
                #     {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
                #     {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
                # ]

                # response = gemini_model.generate_content(prompt, safety_settings=safety_settings)
                response = gemini_model.generate_content(prompt)

                # Check for safety blocks or empty response
                # Accessing parts might raise an error if blocked, check prompt_feedback first
                if response.prompt_feedback and response.prompt_feedback.block_reason:
                     block_reason = response.prompt_feedback.block_reason
                     gemini_summary = f"[Gemini response blocked due to safety filters: {block_reason}]"
                     print(f"Gemini Warning: Blocked - {block_reason}")
                elif not response.parts:
                     # Sometimes no parts even if not explicitly blocked (rare)
                     gemini_summary = "[Gemini response was empty or incomplete]"
                     print(f"Gemini Warning: Response empty. Feedback: {response.prompt_feedback}")
                else:
                    gemini_summary = response.text
                    print("Gemini Summarization Successful")

            except Exception as e:
                print(f"Error calling Gemini API: {e}")
                # Check if the error response from Gemini has more details
                error_details = str(e)
                if hasattr(e, 'response') and hasattr(e.response, 'text'):
                    error_details += f" | Response: {e.response.text}"
                gemini_summary = f"[Error during Gemini API call: {error_details}]"
                traceback.print_exc() # Print full traceback for debugging
        else:
            print("Gemini API not configured. Skipping summarization.")


        # --- Step 5: Return Results ---
        return jsonify({
            "title": submission.title,
            "original_url": original_url, # Keep original URL
            "resolved_url": resolved_url, # Include resolved URL
            "body": submission.selftext,
            "formatted_comments_display": formatted_comments_for_llm,
            "gemini_summary": gemini_summary
        })

    # More specific PRAW exceptions
    except prawcore.exceptions.Redirect:
         print(f"PRAW Error: Redirect encountered for {resolved_url}. Should have been handled by resolver.")
         return jsonify({"error": "Unexpected redirect error after URL resolution."}), 500
    except prawcore.exceptions.NotFound:
         print(f"PRAW Error: Submission not found at {resolved_url}.")
         return jsonify({"error": "Reddit post not found at the resolved URL."}), 404
    except prawcore.exceptions.Forbidden:
         print(f"PRAW Error: Access forbidden for {resolved_url} (private subreddit?).")
         return jsonify({"error": "Access denied to the Reddit post (possibly private or deleted)."}), 403
    except praw.exceptions.InvalidURL: # Should be caught by resolver, but keep as fallback
         return jsonify({"error": "Invalid Reddit URL format after resolution."}), 400
    except praw.exceptions.RedditAPIException as e:
        print(f"Reddit API Error: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Reddit API error: {e}"}), 500
    except requests.exceptions.ConnectionError as e: # Catch potential connection errors during PRAW calls too
        print(f"Connection Error: {e}")
        traceback.print_exc()
        return jsonify({"error": "Network error connecting to Reddit or Gemini."}), 503
    except Exception as e:
        print(f"An unexpected error occurred in /summarize: {e}")
        traceback.print_exc() # Print detailed traceback to Flask console
        return jsonify({"error": f"An unexpected server error occurred: {e}"}), 500

# --- Function to Run Flask ---
def run_flask():
    # Use '127.0.0.1' for local development
    app.run(host='127.0.0.1', port=5000) # Running on default Flask dev server

# --- Start Flask in a Separate Thread ---
# Ensure Flask starts only once
if 'flask_thread' not in st.session_state:
    flask_thread = Thread(target=run_flask, daemon=True)
    st.session_state.flask_thread = flask_thread # Store thread in session state
    flask_thread.start()
    print("Flask thread started.")

# ======== Streamlit Frontend ========

st.title("Reddit Discussion Summarizer")
st.write("Enter a Reddit post link to get a summary.")

# Initialize session state variables if they don't exist
if 'summary_data' not in st.session_state:
    st.session_state.summary_data = None
if 'error_message' not in st.session_state:
    st.session_state.error_message = None
if 'reddit_url_input' not in st.session_state: # Needed to control input field text
     st.session_state.reddit_url_input = ""

# --- Input Area ---
col1, col2, col3 = st.columns([4, 1, 1]) # Adjust columns for buttons
with col1:
    # Use st.session_state to control the input field value
    user_input_url = st.text_input(
        "Reddit Post Link:",
        key="reddit_url_input_widget", # Use a different key for the widget itself
        value=st.session_state.reddit_url_input, # Bind value to session state
        on_change=lambda: setattr(st.session_state, 'reddit_url_input', st.session_state.reddit_url_input_widget) # Update state on change
        )
with col2:
    st.write("") # Spacer
    st.write("") # Spacer
    summarize_button = st.button("Summarize", key="summarize_btn")
with col3:
    st.write("") # Spacer
    st.write("") # Spacer
    clear_button = st.button("Clear", key="clear_btn")

# --- Clear Button Logic ---
if clear_button:
    st.session_state.reddit_url_input = ""  # Clear the stored input value
    st.session_state.summary_data = None      # Clear results
    st.session_state.error_message = None     # Clear errors
    # Explicitly trigger a rerun to update the UI, including the text input
    st.rerun()
    # However, explicitly clearing the widget state might be needed if the binding isn't perfect
    # st.experimental_rerun() # Might cause issues with text_input state, often better to let natural rerun happen

# --- Summarize Button Logic ---
if summarize_button and st.session_state.reddit_url_input:
    input_url = st.session_state.reddit_url_input
    # Basic client-side check (optional, backend does full check)
    if "reddit.com" not in input_url:
        st.session_state.error_message = "Please enter a valid Reddit link (e.g., https://www.reddit.com/...)."
        st.session_state.summary_data = None
        # st.experimental_rerun() # Rerun to show error
    else:
        try:
            backend_url = "http://127.0.0.1:5000/summarize"
            # Use a spinner while waiting for the backend
            with st.spinner(f"Resolving URL, fetching Reddit data, and summarizing... Please wait."):
                response = requests.post(backend_url, json={"url": input_url}, timeout=180) # Increase timeout further

            if response.status_code == 200:
                st.session_state.summary_data = response.json()
                st.session_state.error_message = None # Clear previous errors
            else:
                st.session_state.summary_data = None
                try:
                    error_data = response.json()
                    st.session_state.error_message = f"Error from backend (Status {response.status_code}): {error_data.get('error', 'Unknown error')}"
                except requests.exceptions.JSONDecodeError:
                    st.session_state.error_message = f"Error: Received non-JSON response from backend (Status {response.status_code}). Check Flask logs."
                except Exception as e:
                     st.session_state.error_message = f"Error processing backend response: {e}"

        except requests.exceptions.Timeout:
            st.session_state.summary_data = None
            st.session_state.error_message = "Error: The request to the backend timed out. The post might be too large or the server is busy."
        except requests.exceptions.RequestException as e:
            st.session_state.summary_data = None
            st.session_state.error_message = f"Error connecting to the backend API at {backend_url}. Is Flask running? Error: {e}"
        except Exception as e:
            st.session_state.summary_data = None
            st.session_state.error_message = f"An unexpected error occurred in the Streamlit frontend: {e}"
            st.error(traceback.format_exc()) # Show detailed error in UI for debugging

        st.rerun() # Rerun to display results or error message

elif summarize_button and not st.session_state.reddit_url_input:
    st.warning("Please enter a Reddit link first.")
    st.session_state.summary_data = None
    st.session_state.error_message = None
    # st.experimental_rerun()

# --- Display Area (reads from session state) ---
if st.session_state.error_message:
    st.error(st.session_state.error_message)

if st.session_state.summary_data:
    data = st.session_state.summary_data
    # --- Main Content Area (Title, Body, Summary) ---
    st.subheader(f"Post Title: {data.get('title', 'N/A')}")
    original_link = data.get('original_url','#')
    resolved_link = data.get('resolved_url', '#')
    if original_link != resolved_link and resolved_link != '#':
         st.markdown(f"[Original Link]({original_link}) | [Resolved Link]({resolved_link})")
    else:
         st.markdown(f"[Link to Post]({resolved_link})")


    st.subheader("Original Post Body")
    st.markdown(data.get('body', '*No body text*')) # Use markdown for better formatting

    st.divider() # Visual separator

    st.subheader("✨ Comments Summary ✨")
    st.markdown(data.get('gemini_summary', '*No summary generated.*'))

    # --- Sidebar (Formatted Comments) ---
    with st.sidebar:
        st.subheader("Processed Comments Preview")
        # st.write("Filtered/formatted comment threads used for summarization.")
        # Use text_area for scrollable view
        st.text_area(
            "Comment Threads",
            data.get('formatted_comments_display', '*No comments processed or found.*'),
            height=600, # Adjust height as needed
            key="comments_display_area" # Keep key for potential future interaction
        )


# Add some footer or info text
st.sidebar.markdown("---")
st.sidebar.info("App created by [Harsh Yadav](https://github.com/harshyadav1508)")

# --- END OF FILE app.py ---
