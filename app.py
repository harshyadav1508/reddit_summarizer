# --- START OF FILE app.py ---

import streamlit as st
import requests
import praw
import prawcore # Import for more specific PRAW exceptions
import os
from dotenv import load_dotenv
import google.generativeai as genai # Import Gemini library
import traceback # For detailed error logging
import numpy as np # For FAISS
import faiss       # For vector search
import re          # For splitting text

st.set_page_config(layout="wide") # Use wider layout

# Load environment variables
load_dotenv()

# --- Helper Function for Text Splitting (Unchanged) ---
def split_text_into_chunks(text, max_chunk_size=500, overlap=50):
    """Splits text into potentially overlapping chunks."""
    if not text:
        return []
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks = []
    for para in paragraphs:
        words = para.split()
        current_chunk_words = []
        for word in words:
            current_chunk_words.append(word)
            if len(current_chunk_words) > max_chunk_size * 0.8:
                 chunk_text = " ".join(current_chunk_words)
                 if len(chunk_text) > 10:
                     chunks.append(chunk_text)
                 current_chunk_words = current_chunk_words[-overlap:] if overlap > 0 else []
        if current_chunk_words:
             chunk_text = " ".join(current_chunk_words)
             if len(chunk_text) > 10:
                chunks.append(chunk_text)
    if not chunks and text:
         for i in range(0, len(text), max_chunk_size - overlap):
             chunk = text[i:i + max_chunk_size]
             if len(chunk.strip()) > 10:
                 chunks.append(chunk.strip())
    print(f"Split text into {len(chunks)} chunks.")
    return chunks

# --- Reddit API Setup (Unchanged) ---
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
        print("Reddit API Client Initialized (Read-Only Check Passed)")
        st.session_state.reddit_error = None
    except Exception as e:
        print(f"Error initializing Reddit API: {e}")
        traceback.print_exc()
        st.session_state.reddit_client = None
        st.session_state.reddit_error = f"Failed to initialize Reddit API. Check credentials/logs. Error: {e}"
        st.error(st.session_state.reddit_error)

# --- Gemini API Setup (Unchanged) ---
if 'gemini_model' not in st.session_state:
    try:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        genai.configure(api_key=gemini_api_key)
        st.session_state.gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')
        st.session_state.gemini_embedding_model = 'models/embedding-001'
        print(f"Gemini API Configured Successfully (Model: gemini-1.5-flash-latest, Embeddings: {st.session_state.gemini_embedding_model})")
        st.session_state.gemini_error = None
    except Exception as e:
        print(f"Error initializing Gemini API: {e}")
        traceback.print_exc()
        st.session_state.gemini_model = None
        st.session_state.gemini_embedding_model = None
        st.session_state.gemini_error = f"Failed to configure Gemini API. Summarization/QA will not work. Error: {e}"
        st.warning(st.session_state.gemini_error)


# --- Comment Processing Functions (Unchanged) ---
# format_post_for_llm(...)
# process_comment_recursive(...)
# resolve_reddit_url(...)
# (Keep your existing functions here - unchanged)
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


# --- Core Summarization Logic Function (Unchanged) ---
# get_reddit_summary(...)
# (Keep your existing function here - unchanged)
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
                    # Ensure response.text is available and not None
                    gemini_summary = response.text if hasattr(response, 'text') else "[Gemini response format unexpected]"
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

    # --- Specific Error Handling (Unchanged) ---
    except prawcore.exceptions.Redirect as e:
         print(f"PRAW Error: Unexpected redirect for {resolved_url}. URL: {e.response.url}")
         raise ConnectionError(f"Unexpected redirect error after URL resolution: {resolved_url}") from e
    except prawcore.exceptions.NotFound:
         print(f"PRAW Error: Submission not found at {resolved_url}.")
         raise FileNotFoundError(f"Reddit post not found at the resolved URL: {resolved_url}") # Use more specific built-in exception
    except prawcore.exceptions.Forbidden as e:
         print(f"PRAW Error: Access forbidden for {resolved_url} (private/deleted?).")
         error_msg = f"Access denied to the Reddit post ({resolved_url}). It might be private, quarantined, or deleted."
         if hasattr(e, 'response') and e.response is not None:
             error_msg += f" (HTTP Status: {e.response.status_code})"
         raise PermissionError(error_msg) from e
    except praw.exceptions.InvalidURL: # Should be caught by resolver, but keep as fallback
         raise ValueError(f"Invalid Reddit URL format after resolution: {resolved_url}") # Use ValueError
    except praw.exceptions.RedditAPIException as e:
        print(f"Reddit API Error during fetch/processing: {e}")
        traceback.print_exc()
        raise ConnectionError(f"A Reddit API error occurred: {e}") from e
    except requests.exceptions.ConnectionError as e: # Catch potential connection errors during PRAW calls too
        print(f"Network Connection Error: {e}")
        traceback.print_exc()
        raise ConnectionError(f"Network error connecting to Reddit or Gemini: {e}") from e
    except requests.exceptions.Timeout as e: # Catch timeouts specifically
        print(f"Network Timeout Error: {e}")
        traceback.print_exc()
        raise TimeoutError(f"The request timed out while contacting Reddit or Gemini.") from e
    except Exception as e:
        print(f"An unexpected error occurred in get_reddit_summary: {e}")
        traceback.print_exc() # Print detailed traceback to console/log
        raise RuntimeError(f"An unexpected error occurred during processing: {e}") from e


# --- FAISS Indexing Function (Unchanged) ---
def create_faiss_index(text_chunks, embedding_model_name):
    """Creates a FAISS index from text chunks."""
    if not text_chunks:
        return None, None
    try:
        print(f"Generating embeddings for {len(text_chunks)} chunks using {embedding_model_name}...")
        result = genai.embed_content(
            model=embedding_model_name,
            content=text_chunks,
            task_type="retrieval_document"
        )
        embeddings = result['embedding']
        print(f"Successfully generated {len(embeddings)} embeddings.")
        embeddings_np = np.array(embeddings, dtype='float32')
        if embeddings_np.ndim == 1:
             embeddings_np = np.expand_dims(embeddings_np, axis=0)
        if embeddings_np.ndim != 2 or embeddings_np.shape[0] != len(text_chunks):
            raise ValueError(f"Embeddings shape mismatch: expected ({len(text_chunks)}, D), got {embeddings_np.shape}")
        dimension = embeddings_np.shape[1]
        print(f"Embedding dimension: {dimension}")
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_np)
        print(f"FAISS index created successfully with {index.ntotal} vectors.")
        return index, text_chunks
    except Exception as e:
        print(f"Error creating FAISS index: {e}")
        traceback.print_exc()
        st.error(f"Failed to create search index for the summary: {e}")
        return None, None

# --- RAG Query Function (Unchanged) ---
def query_faiss_and_generate(question, index, chunks, llm_model, embedding_model_name, top_k=3):
    """Queries FAISS, gets context, and generates answer using Gemini."""
    if not index or not chunks:
        return "[Error: Search index not available for Q&A]"
    if not llm_model or not embedding_model_name:
        return "[Error: LLM or embedding model not available for Q&A]"
    try:
        print(f"Embedding question: '{question[:50]}...' using {embedding_model_name}")
        response = genai.embed_content(
            model=embedding_model_name,
            content=question,
            task_type="retrieval_query"
        )
        question_embedding = np.array([response['embedding']], dtype='float32')
        print("Question embedded successfully.")
        print(f"Searching FAISS index for top {top_k} results...")
        distances, indices = index.search(question_embedding, top_k)
        print(f"FAISS search completed. Found indices: {indices[0]}")
        retrieved_chunks = [chunks[i] for i in indices[0] if i < len(chunks)]
        if not retrieved_chunks:
            print("No relevant chunks found in FAISS search.")
            return "I couldn't find specific information about that in the summary."
        context = "\n\n---\n\n".join(retrieved_chunks)
        print(f"Retrieved context (first 100 chars): {context[:100]}...")
        rag_prompt = (
            f"**Instruction:** Answer the following question based *only* on the provided context from the Reddit summary.\n\n"
            f"**Context from Summary:**\n{context}\n\n"
            f"**Question:** {question}\n\n"
            f"**Answer:**"
        )
        print("Generating answer using Gemini with retrieved context...")
        generation_response = llm_model.generate_content(rag_prompt)
        if generation_response.prompt_feedback and generation_response.prompt_feedback.block_reason:
            block_reason = generation_response.prompt_feedback.block_reason
            print(f"Gemini RAG response blocked: {block_reason}")
            return f"[Answer generation blocked due to safety filters: {block_reason}]"
        elif not generation_response.parts:
             print(f"Gemini RAG response empty. Feedback: {generation_response.prompt_feedback}")
             return "[Answer generation failed or was empty]"
        else:
             final_answer = generation_response.text
             print("Gemini RAG answer generated successfully.")
             return final_answer
    except Exception as e:
        print(f"Error during FAISS query or RAG generation: {e}")
        traceback.print_exc()
        return f"[Error answering question: {e}]"


# ======== Streamlit Frontend ========

st.title("Reddit Discussion Summarizer & Q&A")
st.write("Enter a Reddit post link to get a summary, then ask questions about it.")

# Initialize session state variables
if 'summary_data' not in st.session_state:
    st.session_state.summary_data = None
if 'error_message' not in st.session_state:
    st.session_state.error_message = None
if 'reddit_url_input' not in st.session_state:
     st.session_state.reddit_url_input = ""
if 'faiss_index' not in st.session_state:
    st.session_state.faiss_index = None
if 'summary_chunks' not in st.session_state:
    st.session_state.summary_chunks = None
# --- CHANGE: Use chat_history instead of single question/answer ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [] # Store {'role': 'user'/'assistant', 'content': 'message'}

# Display API initialization errors if they occurred
if st.session_state.get('reddit_error'):
    st.error(f"Reddit Initialization Error: {st.session_state.reddit_error}")
if st.session_state.get('gemini_error'):
    st.warning(f"Gemini Initialization Warning: {st.session_state.gemini_error}")


# --- Input Area ---
col1, col2, col3 = st.columns([4, 1, 1])
with col1:
    user_input_url = st.text_input(
        "Reddit Post Link:",
        key="reddit_url_input_widget",
        value=st.session_state.reddit_url_input,
        on_change=lambda: setattr(st.session_state, 'reddit_url_input', st.session_state.reddit_url_input_widget)
        )
with col2:
    st.write("")
    st.write("")
    summarize_button = st.button("Summarize", key="summarize_btn", type="primary")
with col3:
    st.write("")
    st.write("")
    clear_button = st.button("Clear All", key="clear_btn")

# --- Clear Button Logic ---
if clear_button:
    st.session_state.reddit_url_input = ""
    st.session_state.summary_data = None
    st.session_state.error_message = None
    st.session_state.faiss_index = None
    st.session_state.summary_chunks = None
    st.session_state.chat_history = [] # Clear chat history
    print("Session state cleared.")
    st.rerun()

# --- Summarize Button Logic ---
if summarize_button and st.session_state.reddit_url_input:
    input_url = st.session_state.reddit_url_input.strip()
    # Clear previous results AND CHAT before starting new summary
    st.session_state.error_message = None
    st.session_state.summary_data = None
    st.session_state.faiss_index = None
    st.session_state.summary_chunks = None
    st.session_state.chat_history = [] # Clear chat on new summary

    # --- Validation and Summary Generation (Mostly unchanged) ---
    if "reddit.com" not in input_url or not input_url.startswith("http"):
        st.session_state.error_message = "Please enter a valid Reddit link (starting with http:// or https:// and containing reddit.com)."
    elif not st.session_state.get('reddit_client'):
         st.session_state.error_message = "Cannot summarize: Reddit client failed to initialize. Check logs and API keys."
    elif not st.session_state.get('gemini_model') or not st.session_state.get('gemini_embedding_model'):
         st.session_state.error_message = "Cannot summarize/QA: Gemini models failed to initialize. Check logs and API keys."
    else:
        with st.spinner(f"Summarizing '{input_url[:50]}...' Please wait."):
            try:
                summary_result = get_reddit_summary(input_url)
                st.session_state.summary_data = summary_result
                st.session_state.error_message = None
                current_summary = summary_result.get("gemini_summary")
                if current_summary and not current_summary.startswith("["):
                    with st.spinner("Creating searchable index for the summary..."):
                        chunks = split_text_into_chunks(current_summary)
                        if chunks:
                            index, stored_chunks = create_faiss_index(
                                chunks,
                                st.session_state.gemini_embedding_model
                            )
                            if index and stored_chunks:
                                st.session_state.faiss_index = index
                                st.session_state.summary_chunks = stored_chunks
                                print("Summary indexed successfully.")
                            else:
                                st.warning("Could not create a searchable index for the summary. Q&A will be disabled.")
                        else:
                            st.warning("Summary content was empty or too short to index. Q&A will be disabled.")
                elif current_summary:
                    st.warning(f"Summary generated, but may contain errors: {current_summary[:100]}... Q&A disabled.")
                else:
                     st.warning("Summary generation failed. Q&A disabled.")
            except (ValueError, FileNotFoundError, PermissionError, ConnectionError, TimeoutError, RuntimeError) as e:
                st.session_state.summary_data = None
                st.session_state.error_message = f"Error during summarization: {e}"
            except Exception as e:
                st.session_state.summary_data = None
                st.session_state.error_message = f"An unexpected error occurred: {e}"
                st.error(traceback.format_exc())
    st.rerun()

elif summarize_button and not st.session_state.reddit_url_input:
    st.warning("Please enter a Reddit link first.")
    st.session_state.summary_data = None
    st.session_state.error_message = None


# --- Display Area ---
# Removed the explicit columns for main/sidebar

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

    with st.expander("Original Post Body", expanded=False):
        st.markdown(data.get('body', '*No body text*'))

    st.divider()

    st.subheader("✨ Comments Summary ✨")
    summary_text = data.get('gemini_summary', '*No summary generated.*')
    if "[Error" in summary_text or "[Gemini response blocked" in summary_text or "[Summarization skipped" in summary_text:
        st.warning(summary_text)
    else:
        st.markdown(summary_text)

    st.divider()

    # --- CHANGE: Chat Q&A Section ---
    if st.session_state.get('faiss_index') and st.session_state.get('summary_chunks'):
        st.subheader("❓ Chat About the Summary")

        # Display existing chat messages
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # React to user input using st.chat_input
        if prompt := st.chat_input("Ask a question about the summary..."):
            # Add user message to history and display it
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = query_faiss_and_generate(
                            question=prompt,
                            index=st.session_state.faiss_index,
                            chunks=st.session_state.summary_chunks,
                            llm_model=st.session_state.gemini_model,
                            embedding_model_name=st.session_state.gemini_embedding_model,
                            top_k=3
                        )
                        st.markdown(response)
                        # Add assistant response to history
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
                    except Exception as e:
                        error_msg = f"Sorry, an error occurred while answering: {e}"
                        st.error(error_msg)
                        # Optionally add error message to history as assistant response
                        # st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                        print(f"Error in chat input processing: {e}")
                        traceback.print_exc()

    elif data.get("gemini_summary") and not data.get("gemini_summary", "").startswith("["):
         st.info("Q&A about the summary is disabled (index creation failed or summary too short).")

    st.divider()

    # --- CHANGE: Moved Comments Preview to Main Area ---
    with st.expander("Processed Comments Preview (Used for Summary)", expanded=False):
         st.text_area(
            "Comment Threads", # Changed label slightly
            data.get('formatted_comments_display', '*No comments processed or found.*'),
            height=400, # Reduced height slightly
            key="comments_display_area"
        )


# --- Footer (No change needed, can stay outside main data display area) ---
st.sidebar.markdown("---") # Keep footer in sidebar if you like, or move below main content
st.sidebar.info("App created by [Harsh Yadav](https://github.com/harshyadav1508)")

# --- END OF FILE app.py ---