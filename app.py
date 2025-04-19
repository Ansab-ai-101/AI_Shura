import asyncio
import os
import json
import datetime
import tempfile
from typing import Dict, List, Optional, Any # Added Any
import uuid
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser
import time # For potential retries


import streamlit as st
import pandas as pd
import ollama
from duckduckgo_search import DDGS
# Removed: import chromadb
# Removed: from chromadb.config import Settings
# Removed: from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
import lancedb # Added
import pyarrow as pa # Added for schema definition
import numpy as np # Added for vector similarity if needed manually
import requests # Added for standalone Ollama embedding function

from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from crawl4ai.content_filter_strategy import BM25ContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

# --- System Prompts (Remain the same) ---
EMOTION_ANALYSIS_PROMPT = """...""" # Keep as is
GROWTH_PLAN_PROMPT = """...""" # Keep as is
RESOURCE_SYNTHESIS_PROMPT = """...""" # Keep as is
COMMUNITY_SUGGESTION_PROMPT = """...""" # Keep as is

# --- Database Functions (Session State - Remain the same) ---
def initialize_databases():
    """Initialize the databases for storing user data, emotions, and community content."""
    # Use Streamlit session state for simple in-memory storage
    if 'user_db' not in st.session_state:
        st.session_state.user_db = {}
    if 'emotion_db' not in st.session_state:
        st.session_state.emotion_db = {}
    if 'community_db' not in st.session_state:
        st.session_state.community_db = []
    if 'growth_plans_db' not in st.session_state:
        st.session_state.growth_plans_db = {}
    if 'resource_db' not in st.session_state:
        st.session_state.resource_db = {}

def save_user_data(user_id: str, data: Dict):
    """Save user data to the session state database."""
    if 'user_db' in st.session_state:
        st.session_state.user_db[user_id] = data

def save_emotion_entry(user_id: str, emotion_data: Dict):
    """Save an emotion entry to the session state database."""
    if user_id not in st.session_state.emotion_db:
        st.session_state.emotion_db[user_id] = []
    emotion_data['timestamp'] = datetime.datetime.now().isoformat()
    emotion_data['id'] = str(uuid.uuid4())
    st.session_state.emotion_db[user_id].append(emotion_data)
    return emotion_data['id']

# --- Functions for Growth Plans, Community, etc. (Remain the same) ---
def save_growth_plan(user_id: str, emotion_id: str, plan_data: Dict):
     if user_id not in st.session_state.growth_plans_db:
         st.session_state.growth_plans_db[user_id] = {}
     st.session_state.growth_plans_db[user_id][emotion_id] = plan_data

def get_user_emotion_history(user_id: str) -> List[Dict]:
     if 'emotion_db' not in st.session_state or user_id not in st.session_state.emotion_db:
         return []
     return st.session_state.emotion_db[user_id]

def get_emotion_entry(user_id: str, emotion_id: str) -> Optional[Dict]:
     history = get_user_emotion_history(user_id)
     for entry in history:
         if entry.get('id') == emotion_id:
             return entry
     return None

def get_growth_plan(user_id: str, emotion_id: str) -> Optional[Dict]:
     if 'growth_plans_db' not in st.session_state or user_id not in st.session_state.growth_plans_db:
         return None
     return st.session_state.growth_plans_db[user_id].get(emotion_id)

def save_community_post(user_id: str, post_data: Dict):
     post_data['user_id'] = user_id
     post_data['timestamp'] = datetime.datetime.now().isoformat()
     post_data['id'] = str(uuid.uuid4())
     post_data['likes'] = 0
     post_data['comments'] = []
     if 'community_db' not in st.session_state:
         st.session_state.community_db = []
     st.session_state.community_db.append(post_data)
     return post_data['id']

def get_community_posts(limit: int = 20) -> List[Dict]:
     if 'community_db' not in st.session_state:
         return []
     posts = sorted(st.session_state.community_db, key=lambda x: x['timestamp'], reverse=True)
     return posts[:limit]

def add_comment_to_post(post_id: str, user_id: str, comment: str):
      if 'community_db' in st.session_state:
          for post in st.session_state.community_db:
              if post['id'] == post_id:
                  if 'comments' not in post: post['comments'] = []
                  post['comments'].append({'user_id': user_id, 'comment': comment, 'timestamp': datetime.datetime.now().isoformat(), 'id': str(uuid.uuid4())})
                  break

def like_post(post_id: str, user_id: str):
     if 'community_db' in st.session_state:
         for post in st.session_state.community_db:
             if post['id'] == post_id:
                 post['likes'] = post.get('likes', 0) + 1
                 break

def save_resource(user_id: str, emotion_id: str, resource_data: Dict):
     if user_id not in st.session_state.resource_db:
         st.session_state.resource_db[user_id] = {}
     st.session_state.resource_db[user_id][emotion_id] = resource_data

def get_resources(user_id: str, emotion_id: str) -> Optional[Dict]:
     if 'resource_db' not in st.session_state or user_id not in st.session_state.resource_db:
         return None
     return st.session_state.resource_db[user_id].get(emotion_id)


# --- Standalone Ollama Embedding Function ---
@st.cache_data(ttl=3600) # Cache embeddings for an hour
def get_ollama_embeddings(texts: List[str], model_name: str = "nomic-embed-text:latest") -> List[List[float]]:
    """Gets embeddings for a list of texts using a running Ollama instance."""
    ollama_api_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") + "/api/embeddings"
    embeddings = []
    max_retries = 3
    retry_delay = 2 # seconds

    for text in texts:
        if not text or not isinstance(text, str): # Handle empty or non-string inputs
            print(f"Warning: Skipping embedding for invalid input: {text}")
            embeddings.append([]) # Or handle as needed, maybe add a zero vector of correct dimension?
            continue

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    ollama_api_url,
                    json={"model": model_name, "prompt": text},
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                result = response.json()
                if "embedding" in result:
                    embeddings.append(result["embedding"])
                    break # Success, move to next text
                else:
                    print(f"Warning: 'embedding' key not found in Ollama response for text: {text[:50]}...")
                    if attempt == max_retries - 1:
                        embeddings.append([]) # Failed after retries
            except requests.exceptions.RequestException as e:
                print(f"Error calling Ollama embedding API (Attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    st.error(f"Failed to get embedding from Ollama after {max_retries} attempts: {e}")
                    embeddings.append([]) # Failed after retries
            except Exception as e: # Catch other potential errors like JSON decoding
                 print(f"Unexpected error during embedding (Attempt {attempt+1}/{max_retries}): {e}")
                 if attempt == max_retries - 1:
                     embeddings.append([]) # Failed after retries


    # Basic check for dimension consistency (assuming first successful embedding is representative)
    valid_embeddings = [emb for emb in embeddings if emb]
    if not valid_embeddings:
         print("Warning: No valid embeddings were generated.")
         # Determine expected dimension (hardcoded or from model info if possible)
         # For now, return potentially empty lists if all failed
         return embeddings


    expected_dim = len(valid_embeddings[0])
    final_embeddings = []
    for emb in embeddings:
        if len(emb) == expected_dim:
            final_embeddings.append(emb)
        else:
            print(f"Warning: Generated embedding has incorrect dimension ({len(emb)} vs {expected_dim}). Replacing with zeros.")
            final_embeddings.append([0.0] * expected_dim) # Replace malformed/missing with zero vector

    return final_embeddings


# --- LanceDB Integration ---
LANCEDB_URI = "./lancedb_data" # Directory to store LanceDB data

def get_lancedb_table(table_name="web_resources") -> Optional[Any]: # Changed return type hint
    """Creates or opens a LanceDB table."""
    try:
        db = lancedb.connect(LANCEDB_URI)
        table_names = db.table_names()

        if table_name in table_names:
            print(f"Opening existing LanceDB table: {table_name}")
            return db.open_table(table_name)
        else:
            print(f"Creating new LanceDB table: {table_name}")
            # Define a schema - IMPORTANT: Match vector dimension from Ollama model
            # nomic-embed-text default dimension is 768
            # Use `ollama show <model_name> --json` to check dimension if unsure
            vector_dim = 768 # Adjust if using a different Ollama embedding model
            schema = pa.schema(
                [
                    pa.field("id", pa.string()), # Unique ID for each chunk
                    pa.field("text", pa.string()),
                    pa.field("vector", pa.list_(pa.float32(), list_size=vector_dim)),
                    pa.field("source", pa.string()), # URL
                    pa.field("title", pa.string()) # Page title
                ]
            )
            return db.create_table(table_name, schema=schema, mode="overwrite") # Use "create" mode initially

    except Exception as e:
        st.error(f"Error connecting to or creating LanceDB table '{table_name}': {e}")
        print(f"LanceDB Error: {e}")
        return None


def normalize_id(text: str) -> str:
     """Simple normalization for IDs if needed, less critical than for ChromaDB."""
     return text.replace(" ", "_").replace("/", "_").replace(":", "_").replace(".", "_")[:100]


def add_to_vector_database(results, table_name="web_resources"):
    """Adds crawl results to a LanceDB table."""
    table = get_lancedb_table(table_name)
    if not table:
        st.error("LanceDB table not available. Cannot add resources.")
        return []

    data_to_add = []
    texts_to_embed = []
    metadata_list = [] # Store metadata corresponding to texts_to_embed

    # --- Prepare data and collect texts for batch embedding ---
    for result in results:
        if not result or not result.markdown_v2 or not result.markdown_v2.fit_markdown:
            continue
        content = result.markdown_v2.fit_markdown
        if not content.strip():
            continue

        # Simple chunking (same as before)
        chunks = []
        paragraphs = content.split('\n\n')
        for para in paragraphs:
            if len(para) < 50: continue
            if len(para) <= 1000:
                 if para.strip(): chunks.append(para.strip())
            else:
                for i in range(0, len(para), 800):
                    chunk = para[i:i+1000].strip()
                    if chunk: chunks.append(chunk)

        url = result.url or "unknown_url"
        title = result.title or "No Title"

        for idx, chunk in enumerate(chunks):
            if chunk:
                # Use URL + chunk index for a unique ID
                doc_id = f"{normalize_id(url)}_{idx}"
                texts_to_embed.append(chunk)
                metadata_list.append({"id": doc_id, "text": chunk, "source": url, "title": title})

    if not texts_to_embed:
        print("No valid text chunks found to embed.")
        return []

    # --- Get embeddings in batches (Ollama might handle batches internally, but requests are sequential here) ---
    print(f"Generating embeddings for {len(texts_to_embed)} text chunks...")
    embeddings = get_ollama_embeddings(texts_to_embed) # Call the standalone function

    # --- Combine metadata with embeddings ---
    successful_embeddings = 0
    for i, meta in enumerate(metadata_list):
        if i < len(embeddings) and embeddings[i]: # Check if embedding exists and is not empty/error
            meta["vector"] = embeddings[i]
            data_to_add.append(meta)
            successful_embeddings += 1
        else:
             print(f"Warning: Skipping data for chunk {i} due to missing or invalid embedding.")

    if not data_to_add:
        st.error("Failed to generate any valid embeddings. Cannot add data to LanceDB.")
        return []

    # --- Add data to LanceDB table ---
    print(f"Adding {len(data_to_add)} entries to LanceDB table '{table_name}'...")
    try:
        # LanceDB prefers Pandas DataFrame or list of dicts
        # df = pd.DataFrame(data_to_add)
        # table.add(df)
        table.add(data_to_add) # Adding list of dicts directly
        print(f"Successfully added {len(data_to_add)} entries.")

        # Check count (optional)
        print(f"Table '{table_name}' now contains {table.count_rows()} rows.")

        # Return the text content of added documents
        return [d['text'] for d in data_to_add]

    except Exception as e:
        st.error(f"Failed to add data to LanceDB table: {e}")
        print(f"LanceDB Add Error: {e}")
        # Optionally print details of data that failed if possible
        # print("Data attempted:", data_to_add[:2]) # Print first few items
        return []


def query_vector_database(query_text: str, n_results: int = 5, table_name="web_resources") -> List[str]:
    """Queries the LanceDB table for relevant content chunks."""
    table = get_lancedb_table(table_name)
    if not table:
        st.error("LanceDB table not available for querying.")
        return []

    try:
        # 1. Get the embedding for the query text
        query_embedding = get_ollama_embeddings([query_text])

        if not query_embedding or not query_embedding[0]:
            st.error("Could not generate embedding for the query text.")
            return []

        # 2. Perform the search using the query vector
        results = table.search(query_embedding[0]) \
                       .limit(n_results) \
                       .select(["text"]) \
                       .to_list() # Returns a list of dictionaries, e.g., [{'text': '...'}]

        # 3. Extract the text from the results
        return [result['text'] for result in results if 'text' in result]

    except Exception as e:
        st.error(f"Error querying LanceDB table: {e}")
        print(f"LanceDB Query Error: {e}")
        return []


# --- LLM Interaction Functions (Remain the same) ---
def call_llm(prompt: str, system_prompt: str, user_input: str) -> Dict:
     """Call LLM with the given system prompt and user input."""
     # ... (keep existing implementation) ...
     messages = [
         {
             "role": "system",
             "content": system_prompt,
         },
         {
             "role": "user",
             "content": f"{prompt}\n\n{user_input}", # Combine prompt and input
         },
     ]
     try:
         response = ollama.chat(model="llama3:8b", messages=messages)
         response_content = response['message']['content']
         try:
             json_start = response_content.find('{')
             json_end = response_content.rfind('}') + 1
             if json_start != -1 and json_end != -1:
                 json_str = response_content[json_start:json_end]
                 return json.loads(json_str)
             else:
                 st.warning("LLM response was not valid JSON. Displaying raw text.")
                 return {"raw_response": response_content}
         except json.JSONDecodeError:
             st.warning("LLM response could not be parsed as JSON. Displaying raw text.")
             return {"raw_response": response_content}
         except Exception as e:
             st.error(f"Error parsing LLM response: {e}")
             return {"raw_response": response_content}
     except Exception as e:
         st.error(f"Error calling LLM: {e}")
         return {"error": f"Failed to get response from LLM: {e}"}

def analyze_emotion(journal_entry: str) -> Dict:
     return call_llm(
         prompt="Analyze this journal entry for emotional content based on the specified JSON format:",
         system_prompt=EMOTION_ANALYSIS_PROMPT,
         user_input=journal_entry
     )

def generate_growth_plan(emotion_analysis: Dict, user_goals: Dict) -> Dict:
     input_data = {
         "emotion_analysis": emotion_analysis,
         "user_goals": user_goals if user_goals else {"general_goal": "Improve emotional regulation and well-being."}
     }
     return call_llm(
         prompt="Create a growth plan based on this emotional analysis and goals, following the specified JSON format:",
         system_prompt=GROWTH_PLAN_PROMPT,
         user_input=json.dumps(input_data, indent=2)
     )

def synthesize_resources(emotion_analysis: Dict, growth_plan: Optional[Dict], web_content: List[str]) -> Dict:
     if not web_content: return {"error": "No web content provided for synthesis."}
     input_data = {
         "emotion_analysis": emotion_analysis,
         "growth_plan": growth_plan if growth_plan else "No specific growth plan available.",
         "web_content": "\n---\n".join(web_content)
     }
     return call_llm(
         prompt="Synthesize these web resources for emotional growth based on the user's profile, following the specified JSON format:",
         system_prompt=RESOURCE_SYNTHESIS_PROMPT,
         user_input=json.dumps(input_data, indent=2)
     )

def get_community_suggestions(emotion_analysis: Dict, growth_plan: Optional[Dict]) -> Dict:
     input_data = {
         "emotion_analysis": emotion_analysis,
         "growth_plan": growth_plan if growth_plan else "No specific growth plan available."
     }
     return call_llm(
         prompt="Suggest community interactions based on this profile, following the specified JSON format:",
         system_prompt=COMMUNITY_SUGGESTION_PROMPT,
         user_input=json.dumps(input_data, indent=2)
     )

# --- Web Search and Crawl Functions (Remain mostly the same) ---
# Need to make sure crawl_webpages and search_and_crawl_resources call the new add_to_vector_database
def get_web_urls(search_term: str, num_results: int = 5) -> List[str]:
    # ... (keep existing implementation) ...
    try:
        enhanced_search = f"{search_term} emotional regulation coping strategies personal development"
        discard_sites = ["youtube.com", "amazon.com", "pinterest.com", "facebook.com", "instagram.com", "twitter.com", "tiktok.com", "reddit.com/r/"]
        for site in discard_sites: enhanced_search += f" -site:{site}"
        print(f"Searching DuckDuckGo for: {enhanced_search}")
        results = DDGS().text(enhanced_search, max_results=num_results * 2)
        urls = [result["href"] for result in results if result.get("href")]
        print(f"Initial URLs found: {urls}")
        filtered_urls = []
        seen_domains = set()
        for url in urls:
            if url.lower().endswith(".pdf"): continue
            domain = urlparse(url).netloc
            if domain not in seen_domains:
                filtered_urls.append(url)
                seen_domains.add(domain)
        print(f"Filtered URLs (before robots check): {filtered_urls[:num_results]}")
        allowed_urls = check_robots_txt(filtered_urls[:num_results])
        print(f"Allowed URLs (after robots check): {allowed_urls}")
        return allowed_urls
    except Exception as e:
        error_msg = f"❌ Failed to fetch results from the web: {str(e)}"
        print(error_msg)
        st.error(error_msg)
        return []

def check_robots_txt(urls: List[str]) -> List[str]:
    # ... (keep existing implementation) ...
    allowed_urls = []
    rp_cache = {}
    for url in urls:
        parsed_url = urlparse(url)
        scheme = parsed_url.scheme
        netloc = parsed_url.netloc
        if not scheme or not netloc: continue
        robots_url = f"{scheme}://{netloc}/robots.txt"
        rp = rp_cache.get(robots_url)
        if rp is None:
            try:
                rp = RobotFileParser(robots_url)
                rp.read()
                rp_cache[robots_url] = rp
                print(f"Read robots.txt for {netloc}")
            except Exception as e:
                print(f"Could not read or parse robots.txt for {netloc}: {e}. Assuming allowed.")
                allowed_urls.append(url)
                rp_cache[robots_url] = "error"
                continue
        elif rp == "error":
             allowed_urls.append(url)
             continue
        try:
            user_agent = "crawl4ai/python"
            if rp.can_fetch(user_agent, url):
                allowed_urls.append(url)
                print(f"Allowed crawling: {url}")
            else:
                print(f"Disallowed by robots.txt: {url}")
        except Exception as e:
            print(f"Error checking robots.txt permission for {url}: {e}. Assuming allowed.")
            allowed_urls.append(url)
    return allowed_urls


async def crawl_webpages(urls: List[str], query: str):
    # ... (keep existing implementation, ensure BM25ContentFilter and CrawlerRunConfig are correct) ...
    if not urls: return []
    bm25_filter = BM25ContentFilter(user_query=query, bm25_threshold=1.0)
    md_generator = DefaultMarkdownGenerator(content_filter=bm25_filter)
    crawler_config = CrawlerRunConfig(
        markdown_generator=md_generator,
        excluded_tags=["nav", "footer", "header", "form", "img", "a", "script", "style", "aside"],
        only_text=True, exclude_social_media_links=True, keep_data_attributes=False,
        cache_mode=CacheMode.BYPASS, remove_overlay_elements=True,
        user_agent="Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)",
        page_timeout=25000, wait_for_network_idle=True, network_idle_timeout=5000,
    )
    browser_config = BrowserConfig(headless=True, text_mode=True, light_mode=True)
    results = []
    print(f"Starting crawl for {len(urls)} URLs...")
    try:
        async with AsyncWebCrawler(config=browser_config) as crawler:
            crawl_results = await crawler.arun_many(urls, config=crawler_config)
            results.extend(crawl_results)
            print(f"Crawling finished. Got {len(results)} results.")
    except Exception as e:
        st.error(f"An error occurred during web crawling: {e}")
        print(f"Crawling error: {e}")
    valid_results = [res for res in results if res and res.markdown_v2 and res.markdown_v2.fit_markdown]
    print(f"Valid crawl results with markdown: {len(valid_results)}")
    return valid_results


async def search_and_crawl_resources(emotion_analysis: Dict):
    """Search the web, crawl pages, and add to LanceDB vector DB."""
    # ... (Query generation remains the same) ...
    emotion = emotion_analysis.get('primary_emotion', 'emotional challenge')
    triggers = emotion_analysis.get('triggers', [])
    growth_opportunities = emotion_analysis.get('growth_opportunities', [])
    search_queries = [f"how to cope with feeling {emotion}", f"strategies for managing {emotion}", f"understanding triggers for {emotion}", f"personal growth after feeling {emotion}"]
    if triggers: search_queries.append(f"dealing with {emotion} triggered by {triggers[0]}")
    if growth_opportunities: search_queries.append(f"{growth_opportunities[0]} techniques")

    all_urls = set()
    url_limit_per_query = 2
    total_url_limit = 6
    for query in search_queries:
        if len(all_urls) >= total_url_limit: break
        urls = get_web_urls(query, num_results=url_limit_per_query)
        for url in urls:
            if len(all_urls) < total_url_limit: all_urls.add(url)
            else: break
    unique_urls = list(all_urls)
    if not unique_urls:
        st.warning("Could not find any relevant web resources.")
        return []

    st.info(f"Found {len(unique_urls)} unique URLs to crawl...")
    crawl_query = f"{emotion} coping strategies {' '.join(triggers)} {' '.join(growth_opportunities)}"
    crawl_results = await crawl_webpages(unique_urls, query=crawl_query)

    if not crawl_results:
        st.warning("Crawling did not yield usable content from the found URLs.")
        return []

    # --- Call the updated add_to_vector_database ---
    st.info("Adding crawled content to resource database (LanceDB)...")
    web_content_chunks = add_to_vector_database(crawl_results, table_name="web_resources") # Use the new function

    if web_content_chunks:
        st.success(f"Successfully processed {len(crawl_results)} web pages and added {len(web_content_chunks)} content chunks to LanceDB.")
    else:
         st.warning("Content was crawled but could not be added to the LanceDB resource database.")

    return web_content_chunks # Return the chunks added


# --- UI Components (Remain the same, but ensure they call the correct functions) ---
# Make sure render_search_resources, render_resources, render_view_resources
# use the LanceDB-based query_vector_database and search_and_crawl_resources

def render_login_page():
    # ... (Keep existing implementation) ...
    st.title("🌱 Emotion to Action")
    st.subheader("Transform Emotional Experiences into Personal Growth")
    st.markdown("""...""")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Login")
        login_username = st.text_input("Username", key="login_username")
        login_password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login"):
            if login_username in st.session_state.user_db:
                if st.session_state.user_db[login_username].get('password') == login_password: # Hashing needed
                    st.session_state.current_user = login_username
                    st.session_state.authenticated = True
                    st.session_state.current_view = "main"
                    st.rerun()
                else: st.error("Invalid password")
            else: st.error("Username not found")
    with col2:
        st.subheader("Sign Up")
        signup_username = st.text_input("Choose Username", key="signup_username")
        signup_password = st.text_input("Choose Password", type="password", key="signup_password")
        signup_confirm = st.text_input("Confirm Password", type="password", key="signup_confirm")
        if st.button("Sign Up"):
            if not signup_username or not signup_password: st.error("Username and password cannot be empty.")
            elif signup_username in st.session_state.user_db: st.error("Username already taken")
            elif signup_password != signup_confirm: st.error("Passwords do not match")
            else:
                st.session_state.user_db[signup_username] = { # Hashing needed
                    'password': signup_password, 'joined_date': datetime.datetime.now().isoformat(),
                    'premium': False, 'streak': 0, 'points': 0, 'goals': {}
                }
                st.success("Account created! You can now login.")


def render_sidebar():
    # ... (Keep existing implementation) ...
    st.sidebar.title("Navigation")
    if 'current_user' in st.session_state and st.session_state.current_user and st.session_state.current_user in st.session_state.user_db:
        user_data = st.session_state.user_db[st.session_state.current_user]
        st.sidebar.write(f"👋 Hello, {st.session_state.current_user}!")
        st.sidebar.write(f"🔥 Streak: {user_data.get('streak', 0)} days")
        st.sidebar.write(f"⭐ Points: {user_data.get('points', 0)}")
        pages = {"Dashboard": "main", "Journal": "journal", "Community": "community", "Resources": "resources", "Profile": "profile"}
        current_page_name = next((name for name, view in pages.items() if st.session_state.get('current_view') == view), "Dashboard")
        selected_page = st.sidebar.radio("Go to:", options=list(pages.keys()), index=list(pages.keys()).index(current_page_name), key="navigation_radio")
        if pages[selected_page] != st.session_state.get('current_view'):
             st.session_state.current_view = pages[selected_page]
             if 'current_emotion_id' in st.session_state: del st.session_state['current_emotion_id']
             if 'current_resource_query' in st.session_state: del st.session_state['current_resource_query']
             st.rerun()
        st.sidebar.divider()
        if st.sidebar.button("Logout"):
            keys_to_clear = ['authenticated', 'current_user', 'current_view', 'current_emotion_id', 'current_resource_query']
            for key in keys_to_clear:
                if key in st.session_state: del st.session_state[key]
            st.rerun()
    else: st.sidebar.info("Please log in or sign up.")

def render_main_dashboard():
    # ... (Keep existing implementation) ...
     st.title("🌱 Your Growth Dashboard")
     user_id = st.session_state.current_user
     user_data = st.session_state.user_db.get(user_id, {})
     emotion_history = get_user_emotion_history(user_id)
     col1, col2 = st.columns([2, 1])
     with col1: st.markdown(f"### Welcome back, {user_id}!\n...") # Shortened for brevity
     with col2:
         if st.button("📝 New Journal Entry"): st.session_state.current_view = "journal"; st.rerun()
         if st.button("👥 Explore Community"): st.session_state.current_view = "community"; st.rerun()
     st.divider(); st.subheader("Recent Emotional Journey")
     # ... (Loop through recent_emotions and display) ...
     st.divider(); st.subheader("Community Highlights")
     # ... (Loop through community_posts and display) ...

def render_journal_page():
    # ... (Keep existing implementation) ...
    st.title("📝 Emotional Journal")
    st.write("Take a moment to reflect...")
    journal_entry = st.text_area("What are you feeling...", height=250, key="journal_entry_input")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Analyze My Emotions", type="primary"):
            # ... (Analysis and saving logic) ...
            pass # Keep as is
    with col2:
        if st.button("Cancel"): st.session_state.current_view = "main"; st.rerun()
    st.divider(); st.subheader("Past Journal Entries")
    # ... (Loop through past_entries and display) ...

def render_emotion_analysis(emotion_id: str):
    # ... (Keep existing implementation) ...
    # Ensure buttons for "Find Resources" navigate to "search_resources"
    st.title("🧠 Emotion Analysis")
    # ... (Display analysis) ...
    st.divider(); st.subheader("Next Steps")
    col1, col2, col3 = st.columns(3)
    with col1: # Growth Plan Button
        growth_plan = get_growth_plan(st.session_state.current_user, emotion_id)
        if growth_plan:
             if st.button("View Growth Plan"): st.session_state.current_view = "growth_plan"; st.rerun()
        else:
            if st.button("💡 Create Growth Plan", type="primary"): st.session_state.current_view = "create_growth_plan"; st.rerun()
    with col2: # Resources Button
        resources = get_resources(st.session_state.current_user, emotion_id) # Check if synthesis exists
        if resources:
            if st.button("View Resources"): st.session_state.current_view = "view_resources"; st.rerun()
        else:
            emotion_data = get_emotion_entry(st.session_state.current_user, emotion_id)
            analysis_valid = emotion_data and 'analysis' in emotion_data and not ("error" in emotion_data['analysis'] or "raw_response" in emotion_data['analysis'])
            if st.button("🔎 Find/Add Resources", disabled=not analysis_valid):
                 st.session_state.current_view = "search_resources"; st.rerun()
    with col3: # Back Button
        if st.button("Back to Journal"):
            st.session_state.current_view = "journal"
            if 'current_emotion_id' in st.session_state: del st.session_state['current_emotion_id']
            st.rerun()

def render_create_growth_plan(emotion_id: str):
    # ... (Keep existing implementation) ...
     st.title("💡 Create Your Growth Plan")
     # ... (Display context, goal inputs, generation logic) ...

def render_growth_plan(emotion_id: str):
    # ... (Keep existing implementation) ...
     st.title("🚀 Your Growth Plan")
     # ... (Display plan details) ...

def render_search_resources(emotion_id: str):
    # ... (Keep existing implementation - it calls the async search_and_crawl_resources) ...
    st.title("🔎 Find Growth Resources")
    # ... (Display context) ...
    if st.button("Start Resource Search", type="primary"):
        with st.spinner("Searching the web and crawling... Please wait."):
            try:
                # --- Async Execution using asyncio ---
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                crawled_chunks = loop.run_until_complete(search_and_crawl_resources(st.session_state.emotion_analysis_context)) # Pass analysis context if needed
                loop.close()

                if crawled_chunks:
                     st.session_state.last_crawled_chunks = crawled_chunks
                     st.success("Web search and crawling complete! Content added.")
                     st.session_state.current_view = "resources"
                     st.rerun()
                else:
                     st.warning("Search completed, but no usable content was found or added.")
                     # Add back button or stay
            except Exception as e:
                 st.error(f"An error occurred during the resource search process: {e}")
                 print(f"Error in search_and_crawl_resources execution: {e}")

    if st.button("Cancel"): st.session_state.current_view = "emotion_analysis"; st.rerun()

def render_resources():
    # ... (Keep existing implementation - calls query_vector_database for synthesis) ...
    st.title("📚 Growth Resources")
    user_id = st.session_state.current_user
    emotion_id = st.session_state.get('current_emotion_id')
    col1, col2 = st.columns([3, 1])
    with col1: # Synthesis Section
        if emotion_id:
            # ... (Check for existing synthesis, button to synthesize) ...
            if st.button("Synthesize Relevant Resources", type="primary", key="synthesize_now"):
                # ... (Spinner, call query_vector_database, synthesize_resources, save_resource) ...
                pass # Keep logic
    with col2: # General Search
        st.subheader("Search All Resources")
        search_query = st.text_input("Enter keywords...", key="resource_search_query")
        if st.button("Search", key="general_resource_search"):
            if search_query:
                 st.session_state.current_resource_query = search_query
                 st.session_state.current_view = "view_resources"
                 st.rerun()
            else: st.warning("Please enter a search term.")
    st.divider()

def render_view_resources():
    # ... (Keep existing implementation - calls query_vector_database for search results) ...
     st.title("📚 View Resources")
     user_id = st.session_state.current_user
     emotion_id = st.session_state.get('current_emotion_id')
     search_query = st.session_state.get('current_resource_query')
     resource_data = None
     display_mode = "synthesis"
     if search_query:
         display_mode = "search"
         st.subheader(f"Search Results for: '{search_query}'")
         with st.spinner("Searching resource database..."):
             resource_data = query_vector_database(search_query, n_results=15) # List of chunks
     elif emotion_id:
         resource_data = get_resources(user_id, emotion_id) # Synthesis dict
         # ... (Handle if resource_data exists) ...
     else: display_mode = "none"

     if display_mode == "synthesis" and resource_data:
         # ... (Display synthesized fields) ...
         pass
     elif display_mode == "search" and resource_data:
          if resource_data:
              st.info(f"Found {len(resource_data)} relevant snippets.")
              for i, doc in enumerate(resource_data):
                  with st.expander(f"Snippet {i+1}", expanded=(i<3)): st.markdown(doc)
          else: st.warning("No relevant resources found matching your search query.")

     st.divider()
     if st.button("Back to Resources Hub"):
         st.session_state.current_view = "resources"; st.rerun()


def render_community_page():
    # ... (Keep existing implementation) ...
     st.title("👥 Community Hub")
     # ... (AI Suggestions, New Post Form, Display Posts Loop) ...

def render_profile_page():
    # ... (Keep existing implementation) ...
     st.title("👤 Your Profile")
     # ... (Display stats, Goals Form, Settings Placeholders) ...


# --- Main Application Logic (Remains the same) ---
def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(layout="wide", page_title="EmotionToAction")

    # Initialize session state databases
    initialize_databases()

    # Initialize session state variables
    if 'authenticated' not in st.session_state: st.session_state.authenticated = False
    if 'current_user' not in st.session_state: st.session_state.current_user = None
    if 'current_view' not in st.session_state: st.session_state.current_view = "login"
    if 'current_emotion_id' not in st.session_state: st.session_state.current_emotion_id = None
    if 'current_resource_query' not in st.session_state: st.session_state.current_resource_query = None
    # Add context holder for async search function if needed
    if 'emotion_analysis_context' not in st.session_state: st.session_state.emotion_analysis_context = None


    # --- Routing ---
    if not st.session_state.authenticated:
        st.session_state.current_view = "login"
        render_login_page()
    else:
        render_sidebar()
        view = st.session_state.current_view
        emotion_id = st.session_state.get('current_emotion_id')

        # Store context if navigating to search page
        if view == "search_resources" and emotion_id:
             emotion_data = get_emotion_entry(st.session_state.current_user, emotion_id)
             if emotion_data and 'analysis' in emotion_data:
                  st.session_state.emotion_analysis_context = emotion_data['analysis']


        # Render the current view
        if view == "main": render_main_dashboard()
        elif view == "journal": render_journal_page()
        elif view == "emotion_analysis":
            if emotion_id: render_emotion_analysis(emotion_id)
            else: st.warning("No emotion entry selected."); st.session_state.current_view = "journal"; st.rerun()
        elif view == "create_growth_plan":
             if emotion_id: render_create_growth_plan(emotion_id)
             else: st.warning("Cannot create plan without selection."); st.session_state.current_view = "journal"; st.rerun()
        elif view == "growth_plan":
             if emotion_id: render_growth_plan(emotion_id)
             else: st.warning("Cannot view plan without selection."); st.session_state.current_view = "journal"; st.rerun()
        elif view == "search_resources":
             # Check if context is ready before rendering
             if st.session_state.emotion_analysis_context:
                 render_search_resources(emotion_id) # emotion_id is still useful here maybe?
             elif emotion_id: # Context not set yet, maybe still loading
                 st.warning("Preparing resource search context...")
                 # Could try re-running or fetching context again here if needed.
                 # For simplicity, just show warning or go back.
                 time.sleep(1) # Basic wait, not ideal
                 st.rerun()
             else:
                  st.warning("Cannot search resources without selection."); st.session_state.current_view = "journal"; st.rerun()
        elif view == "resources": render_resources()
        elif view == "view_resources": render_view_resources()
        elif view == "community": render_community_page()
        elif view == "profile": render_profile_page()
        elif view == "login": render_login_page()
        else: st.warning(f"Unknown view: {view}."); st.session_state.current_view = "main"; render_main_dashboard()


if __name__ == "__main__":
    # Ensure the LanceDB data directory exists
    os.makedirs(LANCEDB_URI, exist_ok=True)
    main()
