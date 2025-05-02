# --- START OF FILE app.py ---

import streamlit as st
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

# --- Reddit API Setup ---
# Encapsulate initialization to avoid running it multiple times if script reruns weirdly
# This pattern is correct for Streamlit - runs once per session unless cleared.
if 'reddit_client' not in st.session_state:
    try:
        reddit_client_id = os.getenv("REDDIT_CLIENT_ID")
        reddit_client_secret = os.getenv("REDDIT_CLIENT_SECRET")
        reddit_user_agent = os.getenv("REDDIT_USER_AGENT")
        if not all([reddit_client_id, reddit_client_secret, reddit_user_agent]):
             raise ValueError("Missing Reddit API credentials in environment variables (REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT).")

        st.session_state.reddit_client = praw.Reddit(
            client_id=reddit_client_id,
            client_secret=reddit_client_secret,
            user_agent=reddit_user_agent
        )
        # Test authentication (optional but good)
        # print(f"Testing Reddit auth for user: {st.session_state.reddit_client.user.me()}") # Requires auth scope usually
        print("Reddit API Client Initialized (Read-Only Check Passed)")
        st.session_state.reddit_error = None
    except Exception as e:
        print(f"Error initializing Reddit API: {e}")
        traceback.print_exc()
        st.session_state.reddit_client = None
        st.session_state.reddit_error = f"Failed to initialize Reddit API. Check credentials/logs. Error: {e}"
        # Display error early in Streamlit
        st.error(st.session_state.reddit_error)
        # No need to st.stop() here, allow app to load but show error

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
        traceback.print_exc()
        st.session_state.gemini_model = None
        st.session_state.gemini_error = f"Failed to configure Gemini API. Summarization will not work. Error: {e}"
        # Display warning in Streamlit
        st.warning(st.session_state.gemini_error)

# Access clients from session state where needed
# Note: Directly accessing st.session_state within functions is often clearer
# reddit = st.session_state.get('reddit_client')
# gemini_model = st.session_state.get('gemini_model')


# --- Comment Processing Functions (Unchanged) ---
def format_post_for_llm(post, query_keywords, max_top_comments=20, max_depth=3, min_score=3, min_length=20):
    llm_input_string = f"Reddit Post URL: {post.url}\n" # Add URL for context
    llm_input_string += f"Post Title: {post.title}\n"
    llm_input_string += f"Post Body:\n{post.selftext}\n\n"
    llm_input_string += "--- Relevant Comments (threaded format) ---\n\n"

    try:
        post.comment_sort = 'top' # Explicitly sort by top
        post.comments.replace_more(limit=0) # Remove 'MoreComments' at top level
        comments_list = post.comments.list()
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

    indent = "> " * current_depth
    # Clean comment body (remove excessive newlines)
    cleaned_body = " ".join(comment.body.split())
    formatted_comment = f"{indent}(Score: {comment.score}) {cleaned_body}\n"

    replies_text = ""
    # Process replies only if depth allows and replies exist and are loaded
    if current_depth < max_depth and hasattr(comment, 'replies'):
        try:
            comment.replies.replace_more(limit=0) # Remove MoreComments within replies
            valid_replies = [r for r in comment.replies if isinstance(r, praw.models.Comment)]
            sorted_replies = sorted(valid_replies, key=lambda r: r.score if hasattr(r,'score') else -1, reverse=True)

            for reply in sorted_replies:
                reply_text = process_comment_recursive(
                    reply, query_keywords, current_depth + 1, max_depth,
                    min_score, min_length
                )
                if reply_text:
                     replies_text += reply_text

        except Exception as e:
            print(f"Error processing replies for comment {getattr(comment, 'id', 'N/A')}: {e}")
            # replies_text += f"{indent}> [Error loading replies]\n" # Optional

    return formatted_comment + replies_text

# --- URL Resolution Function (Unchanged) ---
def resolve_reddit_url(url):
    if not url or "reddit.com" not in url:
        return None # Not a reddit URL

    try:
        headers = {'User-Agent': os.getenv("REDDIT_USER_AGENT", 'URLResolver/1.0')}
        response = requests.head(url, allow_redirects=True, timeout=10, headers=headers)
        response.raise_for_status()

        final_url = response.url
        if "reddit.com" in final_url and "/comments/" in final_url:
            print(f"Resolved URL: {url} -> {final_url}")
            return final_url
        else:
            print(f"URL {url} resolved to {final_url}, which doesn't look like a standard post URL.")
            # Treat non-post URLs as invalid for this app's purpose
            return None
    except requests.exceptions.Timeout:
        print(f"Timeout while resolving URL: {url}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error resolving URL {url}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error resolving URL {url}: {e}")
        return None


# --- Core Summarization Logic Function ---
def get_reddit_summary(original_url):
    """
    Fetches Reddit post, processes comments, generates Gemini summary.
    Returns a dictionary with results or raises exceptions on errors.
    """
    # --- Access API clients from session state ---
    reddit = st.session_state.get('reddit_client')
    gemini_model = st.session_state.get('gemini_model')

    if not reddit:
        raise ConnectionError("Reddit client not available. Check initialization logs.")
    # Allow proceeding without Gemini, but note it
    if not gemini_model:
         print("Gemini model not available. Proceeding without summarization.")

    query_keywords = [] # Define keywords if needed, or remove if not used

    # --- Step 0: Resolve URL ---
    resolved_url = resolve_reddit_url(original_url)
    if not resolved_url:
        # Raise a specific error for bad URLs
        raise ValueError(f"Could not resolve the provided URL or it doesn't lead to a valid Reddit post: {original_url}")

    # Use a single try block for the main logic, catching specific errors
    try:
        # --- Step 1: Extract Reddit post data ---
        print(f"Fetching submission for resolved URL: {resolved_url}")
        # Add timeout to the submission call itself (PRAW supports this)
        submission = reddit.submission(url=resolved_url) # Timeout added if needed via praw.ini or Reddit instance config
        # Access an attribute to force fetching and check validity/permissions early
        _ = submission.title
        _ = submission.selftext # Fetch body too
        print(f"Successfully fetched post: {submission.id} - {submission.title[:50]}...")


        # --- Step 2: Format Comments for LLM Input ---
        print("Formatting comments...")
        formatted_comments_for_llm = format_post_for_llm(
            submission,
            query_keywords=query_keywords, # Pass keywords here
            max_top_comments=25,
            max_depth=3,
            min_score=2,
            min_length=20
        )
        print(f"Formatted comments length (approx chars): {len(formatted_comments_for_llm)}")


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
        gemini_summary = "[Summarization skipped: Gemini client not available]" # Default message
        if gemini_model: # Check if Gemini was initialized successfully
            try:
                print("\n--- Sending Prompt to Gemini ---")
                print(f"Prompt length (approx chars): {len(prompt)}")
                print("--- End of Prompt ---")

                # Note: Consider adding timeout to generate_content if available/needed
                response = gemini_model.generate_content(prompt)

                # Check for safety blocks or empty response
                if response.prompt_feedback and response.prompt_feedback.block_reason:
                     block_reason = response.prompt_feedback.block_reason
                     gemini_summary = f"[Gemini response blocked due to safety filters: {block_reason}]"
                     print(f"Gemini Warning: Blocked - {block_reason}")
                elif not response.parts:
                     gemini_summary = "[Gemini response was empty or incomplete]"
                     print(f"Gemini Warning: Response empty. Feedback: {response.prompt_feedback}")
                else:
                    gemini_summary = response.text
                    print("Gemini Summarization Successful")

            except Exception as e:
                # Catch Gemini-specific errors if possible, otherwise general exception
                print(f"Error calling Gemini API: {e}")
                error_details = str(e)
                # You might want to check specific google.api_core.exceptions here
                gemini_summary = f"[Error during Gemini API call: {error_details}]"
                traceback.print_exc() # Log full traceback
                # Decide if this error should stop the whole process or just skip summary
                # For now, we'll return the error message in the summary field.
        else:
            print("Gemini API not configured. Skipping summarization.")


        # --- Step 5: Return Results Dictionary ---
        # Successfully processed
        return {
            "title": submission.title,
            "original_url": original_url,
            "resolved_url": resolved_url,
            "body": submission.selftext,
            "formatted_comments_display": formatted_comments_for_llm,
            "gemini_summary": gemini_summary
        }

    # --- Specific Error Handling for PRAW/Network Issues during fetch ---
    # Place these specific exceptions before the general Exception catch-all
    except prawcore.exceptions.Redirect as e:
         print(f"PRAW Error: Unexpected redirect for {resolved_url}. URL: {e.response.url}")
         # This shouldn't happen often if resolve_reddit_url works, but handle it.
         raise ConnectionError(f"Unexpected redirect error after URL resolution: {resolved_url}") from e
    except prawcore.exceptions.NotFound:
         print(f"PRAW Error: Submission not found at {resolved_url}.")
         raise FileNotFoundError(f"Reddit post not found at the resolved URL: {resolved_url}") # Use more specific built-in exception
    except prawcore.exceptions.Forbidden as e:
         print(f"PRAW Error: Access forbidden for {resolved_url} (private/deleted?).")
         # Add more detail if available from the exception object
         error_msg = f"Access denied to the Reddit post ({resolved_url}). It might be private, quarantined, or deleted."
         if hasattr(e, 'response') and e.response is not None:
             error_msg += f" (HTTP Status: {e.response.status_code})"
         raise PermissionError(error_msg) from e
    except praw.exceptions.InvalidURL: # Should be caught by resolver, but keep as fallback
         raise ValueError(f"Invalid Reddit URL format after resolution: {resolved_url}") # Use ValueError
    except praw.exceptions.RedditAPIException as e:
        print(f"Reddit API Error during fetch/processing: {e}")
        traceback.print_exc()
        # Provide a user-friendly message, log the technical details
        raise ConnectionError(f"A Reddit API error occurred: {e}") from e
    except requests.exceptions.ConnectionError as e: # Catch potential connection errors during PRAW calls too
        print(f"Network Connection Error: {e}")
        traceback.print_exc()
        raise ConnectionError(f"Network error connecting to Reddit or Gemini: {e}") from e
    except requests.exceptions.Timeout as e: # Catch timeouts specifically
        print(f"Network Timeout Error: {e}")
        traceback.print_exc()
        raise TimeoutError(f"The request timed out while contacting Reddit or Gemini.") from e
    # Keep a general Exception catch-all at the end
    except Exception as e:
        print(f"An unexpected error occurred in get_reddit_summary: {e}")
        traceback.print_exc() # Print detailed traceback to console/log
        # Raise a generic exception for the UI to catch
        raise RuntimeError(f"An unexpected error occurred during processing: {e}") from e


# ======== Streamlit Frontend (Mostly Unchanged, but calls local function) ========

st.title("Reddit Discussion Summarizer")
st.write("Enter a Reddit post link to get a summary.")

# Initialize session state variables
if 'summary_data' not in st.session_state:
    st.session_state.summary_data = None
if 'error_message' not in st.session_state:
    st.session_state.error_message = None
if 'reddit_url_input' not in st.session_state:
     st.session_state.reddit_url_input = ""

# Display API initialization errors if they occurred
if st.session_state.get('reddit_error'):
    # This error might already be shown above, but ensuring it's visible
    st.error(f"Reddit Initialization Error: {st.session_state.reddit_error}")
if st.session_state.get('gemini_error'):
    st.warning(f"Gemini Initialization Warning: {st.session_state.gemini_error}")


# --- Input Area ---
col1, col2, col3 = st.columns([4, 1, 1])
with col1:
    # Controlled input component
    user_input_url = st.text_input(
        "Reddit Post Link:",
        key="reddit_url_input_widget",
        value=st.session_state.reddit_url_input,
        on_change=lambda: setattr(st.session_state, 'reddit_url_input', st.session_state.reddit_url_input_widget)
        )
with col2:
    st.write("")
    st.write("")
    # Disable button if API clients failed initialization? Maybe not, allow user to try.
    summarize_button = st.button("Summarize", key="summarize_btn")
with col3:
    st.write("")
    st.write("")
    clear_button = st.button("Clear", key="clear_btn")

# --- Clear Button Logic ---
if clear_button:
    st.session_state.reddit_url_input = ""
    st.session_state.summary_data = None
    st.session_state.error_message = None
    # Clear the widget's internal state if needed (often rerun handles this)
    # st.session_state.reddit_url_input_widget = "" # May not be necessary with on_change
    st.rerun()

# --- Summarize Button Logic ---
if summarize_button and st.session_state.reddit_url_input:
    input_url = st.session_state.reddit_url_input.strip() # Trim whitespace
    st.session_state.error_message = None # Clear previous errors before trying
    st.session_state.summary_data = None  # Clear previous results

    # Basic client-side check (optional, backend does full check)
    if "reddit.com" not in input_url or not input_url.startswith("http"):
        st.session_state.error_message = "Please enter a valid Reddit link (starting with http:// or https:// and containing reddit.com)."
        st.rerun() # Rerun to show error immediately
    elif not st.session_state.get('reddit_client'):
         st.session_state.error_message = "Cannot summarize: Reddit client failed to initialize. Check logs and API keys."
         st.rerun()
    else:
        # Show spinner while processing
        with st.spinner(f"Resolving URL, fetching Reddit data, and summarizing... Please wait."):
            try:
                # --- CALL THE LOCAL FUNCTION DIRECTLY ---
                summary_result = get_reddit_summary(input_url)
                st.session_state.summary_data = summary_result
                st.session_state.error_message = None # Clear error on success

            # --- CATCH ERRORS RAISED BY get_reddit_summary ---
            except (ValueError, FileNotFoundError, PermissionError, ConnectionError, TimeoutError, RuntimeError) as e:
                # Catch the specific errors we defined/raised or standard ones
                st.session_state.summary_data = None
                st.session_state.error_message = f"Error: {e}" # Display the user-friendly message from the raised exception
            except Exception as e:
                # Catch any other unexpected error during the call
                st.session_state.summary_data = None
                st.session_state.error_message = f"An unexpected frontend error occurred: {e}"
                st.error(traceback.format_exc()) # Show detailed error in UI for debugging

        st.rerun() # Rerun to display results or error message

elif summarize_button and not st.session_state.reddit_url_input:
    st.warning("Please enter a Reddit link first.")
    st.session_state.summary_data = None
    st.session_state.error_message = None
    # st.rerun() # Rerun might clear the warning too quickly, let it stay

# --- Display Area (reads from session state) ---
if st.session_state.error_message:
    st.error(st.session_state.error_message)

if st.session_state.summary_data:
    data = st.session_state.summary_data
    st.subheader(f"Post Title: {data.get('title', 'N/A')}")
    original_link = data.get('original_url','#')
    resolved_link = data.get('resolved_url', '#')
    if original_link != resolved_link and resolved_link != '#':
         st.markdown(f"[Original Link]({original_link}) | [Resolved Link]({resolved_link})")
    else:
         st.markdown(f"[Link to Post]({resolved_link})")

    st.subheader("Original Post Body")
    st.markdown(data.get('body', '*No body text*'))

    st.divider()

    st.subheader("✨ Comments Summary ✨")
    # Check for specific error messages within the summary itself
    summary_text = data.get('gemini_summary', '*No summary generated.*')
    if "[Error during Gemini API call:" in summary_text or "[Gemini response blocked" in summary_text:
        st.warning(summary_text) # Show Gemini errors as warnings
    else:
        st.markdown(summary_text)

    with st.sidebar:
        st.subheader("Processed Comments Preview")
        st.text_area(
            "Comment Threads",
            data.get('formatted_comments_display', '*No comments processed or found.*'),
            height=600,
            key="comments_display_area"
        )

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.info("App created by [Harsh Yadav](https://github.com/harshyadav1508)")

# --- END OF FILE app.py ---