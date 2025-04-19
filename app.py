import asyncio
import os
import json
import datetime
from typing import Dict, List, Optional, Any
import uuid
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser
import time
import copy # For deep copying session state items if needed

import streamlit as st
import pandas as pd # Optional, but can be useful
import ollama
from duckduckgo_search import DDGS
import requests # For potential direct API calls if needed (though ollama client is used)

# crawl4ai requires playwright, ensure it's installed: pip install crawl4ai[playwright]
# and browsers installed: playwright install --with-deps
try:
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
    from crawl4ai.content_filter_strategy import BM25ContentFilter
    from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
    CRAWL4AI_AVAILABLE = True
except ImportError:
    st.error("crawl4ai library not found. Please install it: pip install 'crawl4ai[playwright]' and run 'playwright install --with-deps'")
    CRAWL4AI_AVAILABLE = False


# --- Constants and System Prompts ---
OLLAMA_MODEL = "llama3:8b" # Change if needed

# System prompts (Keep as previously defined)
EMOTION_ANALYSIS_PROMPT = """
You are an AI assistant specializing in emotional analysis and personal growth.
Analyze the emotional content in the user's journal entry, focusing on:
1. Identifying primary emotions (joy, sadness, anger, fear, etc.)
2. Recognizing emotion intensity (1-10 scale)
3. Detecting emotional patterns and triggers
4. Observing potential growth opportunities

Format your response as a JSON object with these keys:
- primary_emotion: The dominant emotion expressed
- intensity: A numerical value from 1-10
- triggers: A list of potential triggers identified
- patterns: Any emotional patterns detected
- growth_opportunities: 3 specific ways the user might grow from this experience
- action_steps: 3 suggested concrete actions the user could take

Base your analysis solely on the provided journal entry. Be empathetic yet objective in your assessment.
"""

GROWTH_PLAN_PROMPT = """
You are an AI coach specializing in transforming emotional experiences into personal growth opportunities.
Based on the user's emotional profile and goals, create a structured growth plan that includes:

1. Short-term actions (next 24-48 hours)
2. Medium-term practices (1-2 weeks)
3. Long-term behavior changes (1-3 months)

Format your response as a JSON object with these keys:
- short_term_actions: List of 3 immediate actions
- medium_term_practices: List of 3 practices to develop over weeks
- long_term_changes: List of 3 behavior patterns to cultivate
- reflection_prompts: List of 3 questions for daily reflection
- success_metrics: List of 3 ways to measure progress

Make all suggestions specific, actionable, and tailored to the user's emotional state and goals.
"""

RESOURCE_SYNTHESIS_PROMPT = """
You are an AI assistant specializing in synthesizing web resources for emotional growth.
Based on the user's emotional state and growth goals, synthesize the provided raw web content (provided as markdown snippets) into actionable resources.

Format your response as a JSON object with these keys:
- key_insights: List of 3-5 most relevant insights from the resources
- practical_exercises: List of 2-3 practical exercises mentioned in the resources
- recommended_readings: List of any specific books, articles, or resources mentioned
- expert_advice: Summary of expert advice found in the resources
- source_urls: List of the original URLs the content was derived from
- action_plan: 3 steps to implement these insights based on the user's emotional state

Maintain a compassionate, supportive tone while focusing on factual, evidence-based information. Extract information only from the provided web content snippets.
"""

COMMUNITY_SUGGESTION_PROMPT = """
You are an AI community facilitator specializing in emotional growth.
Based on the user's emotional profile, goals, and growth plan, suggest relevant community resources:

1. Types of experiences that might benefit from community sharing
2. Community topics that align with user's growth areas
3. Potential community support needs

Format your response as a JSON object with these keys:
- sharing_opportunities: List of 3 aspects of the user's journey that could benefit from sharing
- recommended_topics: List of 3 community discussion topics aligned with user's growth areas
- support_needs: List of 3 ways community members might support this journey

Ensure all suggestions maintain user privacy while facilitating meaningful connections.
"""

# --- Session State Database Functions ---
def initialize_app_state():
    """Initialize session state variables if they don't exist."""
    defaults = {
        'authenticated': False,
        'current_user': None,
        'current_view': "login",
        'current_emotion_id': None, # ID of the emotion entry being worked on
        'selected_resource_emotion_id': None, # ID for viewing specific synthesized resource

        # Simple in-memory databases using session state
        'user_db': {}, # {username: {'password': '', 'joined_date': '', 'premium': False, 'streak': 0, 'points': 0, 'goals': {}}}
        'emotion_db': {}, # {username: [{'id': '', 'timestamp': '', 'journal_entry': '', 'analysis': {}}]}
        'community_db': [], # [{'id': '', 'user_id': '', 'timestamp': '', 'title': '', 'content': '', 'likes': 0, 'comments': []}]
        'growth_plans_db': {}, # {username: {emotion_id: {plan_data}}}
        'resource_synthesis_db': {}, # {username: {emotion_id: {synthesis_json}}}
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# --- User Data ---
def save_user_data(user_id: str, data: Dict):
    if 'user_db' in st.session_state:
        st.session_state.user_db[user_id] = data

def get_user_data(user_id: str) -> Optional[Dict]:
     return st.session_state.user_db.get(user_id)

# --- Emotion Journal Data ---
def save_emotion_entry(user_id: str, journal_entry: str, analysis: Dict):
    if user_id not in st.session_state.emotion_db:
        st.session_state.emotion_db[user_id] = []
    entry = {
        'id': str(uuid.uuid4()),
        'timestamp': datetime.datetime.now().isoformat(),
        'journal_entry': journal_entry,
        'analysis': analysis
    }
    st.session_state.emotion_db[user_id].append(entry)
    return entry['id']

def get_user_emotion_history(user_id: str) -> List[Dict]:
    return st.session_state.emotion_db.get(user_id, [])

def get_emotion_entry(user_id: str, emotion_id: str) -> Optional[Dict]:
    history = get_user_emotion_history(user_id)
    for entry in history:
        if entry.get('id') == emotion_id:
            return entry
    return None

# --- Growth Plan Data ---
def save_growth_plan(user_id: str, emotion_id: str, plan_data: Dict):
    if user_id not in st.session_state.growth_plans_db:
        st.session_state.growth_plans_db[user_id] = {}
    st.session_state.growth_plans_db[user_id][emotion_id] = plan_data

def get_growth_plan(user_id: str, emotion_id: str) -> Optional[Dict]:
    if user_id not in st.session_state.growth_plans_db:
        return None
    return st.session_state.growth_plans_db[user_id].get(emotion_id)

# --- Resource Synthesis Data ---
def save_synthesized_resource(user_id: str, emotion_id: str, synthesis_data: Dict):
    if user_id not in st.session_state.resource_synthesis_db:
        st.session_state.resource_synthesis_db[user_id] = {}
    st.session_state.resource_synthesis_db[user_id][emotion_id] = synthesis_data

def get_synthesized_resource(user_id: str, emotion_id: str) -> Optional[Dict]:
    if user_id not in st.session_state.resource_synthesis_db:
        return None
    return st.session_state.resource_synthesis_db[user_id].get(emotion_id)

def get_all_synthesized_resources(user_id: str) -> Dict[str, Dict]:
     """Gets all synthesized resources for a user, keyed by emotion_id."""
     return st.session_state.resource_synthesis_db.get(user_id, {})

# --- Community Data ---
def save_community_post(user_id: str, title: str, content: str):
    post_data = {
        'id': str(uuid.uuid4()),
        'user_id': user_id,
        'timestamp': datetime.datetime.now().isoformat(),
        'title': title,
        'content': content,
        'likes': 0,
        'comments': [] # format: {'id': '', 'user_id': '', 'comment': '', 'timestamp': ''}
    }
    st.session_state.community_db.append(post_data)
    return post_data['id']

def get_community_posts(limit: int = 20) -> List[Dict]:
    posts = sorted(st.session_state.community_db, key=lambda x: x['timestamp'], reverse=True)
    return posts[:limit]

def add_comment_to_post(post_id: str, user_id: str, comment: str):
    for post in st.session_state.community_db:
        if post['id'] == post_id:
            post['comments'].append({
                'id': str(uuid.uuid4()),
                'user_id': user_id,
                'comment': comment,
                'timestamp': datetime.datetime.now().isoformat()
            })
            break

def like_post(post_id: str):
    for post in st.session_state.community_db:
        if post['id'] == post_id:
            post['likes'] = post.get('likes', 0) + 1
            break

# --- LLM Interaction Functions ---
def call_ollama_chat(system_prompt: str, user_prompt: str) -> Dict:
    """Generic function to call Ollama chat endpoint."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        response = ollama.chat(model=OLLAMA_MODEL, messages=messages)
        response_content = response['message']['content']
        # Attempt to parse JSON strictly from the beginning
        try:
            # Clean potential markdown code block fences
            if response_content.startswith("```json"):
                 response_content = response_content[7:]
            if response_content.endswith("```"):
                 response_content = response_content[:-3]
            response_content = response_content.strip()

            parsed_json = json.loads(response_content)
            return parsed_json
        except json.JSONDecodeError:
            st.warning(f"LLM response for '{system_prompt[:30]}...' was not valid JSON. Returning raw.")
            print(f"RAW LLM Response:\n{response_content}") # Log raw response
            return {"raw_response": response_content}

    except Exception as e:
        st.error(f"Error calling Ollama (Model: {OLLAMA_MODEL}): {e}")
        print(f"Ollama Error: {e}")
        return {"error": str(e)}

# Specific LLM tasks
def analyze_emotion(journal_entry: str) -> Dict:
    return call_ollama_chat(EMOTION_ANALYSIS_PROMPT, journal_entry)

def generate_growth_plan(emotion_analysis: Dict, user_goals: Dict) -> Dict:
    input_data = {"emotion_analysis": emotion_analysis, "user_goals": user_goals}
    return call_ollama_chat(GROWTH_PLAN_PROMPT, json.dumps(input_data, indent=2))

def synthesize_resources(emotion_analysis: Dict, growth_plan: Optional[Dict], web_content_list: List[Dict]) -> Dict:
    """Synthesizes resources from a list of dicts containing url and markdown."""
    if not web_content_list:
        return {"error": "No web content provided for synthesis."}

    # Combine markdown content with source URL markers
    combined_content = ""
    source_urls = []
    for item in web_content_list:
        url = item.get('url', 'Unknown Source')
        markdown = item.get('markdown', '')
        if markdown:
             combined_content += f"\n\n--- Content from {url} ---\n\n{markdown}"
             source_urls.append(url)

    if not combined_content:
         return {"error": "Valid web content was empty after processing."}

    input_data = {
        "emotion_analysis": emotion_analysis,
        "growth_plan": growth_plan if growth_plan else "No specific growth plan available.",
        "web_content_snippets": combined_content,
        "source_urls_provided": list(set(source_urls)) # Pass unique URLs
    }

    synthesis_result = call_ollama_chat(RESOURCE_SYNTHESIS_PROMPT, json.dumps(input_data, indent=2))

    # Add source URLs to the result if not already present by the LLM (as fallback)
    if isinstance(synthesis_result, dict) and 'source_urls' not in synthesis_result:
         synthesis_result['source_urls'] = list(set(source_urls))

    return synthesis_result


def get_community_suggestions(emotion_analysis: Dict, growth_plan: Optional[Dict]) -> Dict:
    input_data = {"emotion_analysis": emotion_analysis, "growth_plan": growth_plan}
    return call_ollama_chat(COMMUNITY_SUGGESTION_PROMPT, json.dumps(input_data, indent=2))


# --- Web Search and Crawl Functions ---
def get_web_urls(search_term: str, num_results: int = 3) -> List[str]:
    """Performs a web search and returns filtered URLs."""
    allowed_urls = []
    try:
        enhanced_search = f"{search_term} emotional regulation coping strategies therapy techniques"
        print(f"Searching DDG for: {enhanced_search}")
        results = DDGS().text(enhanced_search, max_results=num_results * 2) # Fetch slightly more
        urls = [result["href"] for result in results if result.get("href")]

        # Basic filtering (remove common problematic domains, PDFs)
        filtered_urls = []
        seen_domains = set()
        discard_domains = {"youtube.com", "amazon.com", "pinterest.com", "facebook.com", "instagram.com", "twitter.com", "tiktok.com"}
        for url in urls:
            if url.lower().endswith(".pdf"): continue
            try:
                domain = urlparse(url).netloc.replace("www.", "")
                if domain and domain not in seen_domains and domain not in discard_domains:
                    filtered_urls.append(url)
                    seen_domains.add(domain)
            except Exception:
                 print(f"Skipping invalid URL: {url}")


        print(f"Filtered URLs (before robots check): {filtered_urls[:num_results]}")
        allowed_urls = check_robots_txt(filtered_urls[:num_results]) # Limit to desired number
        print(f"Allowed URLs (after robots check): {allowed_urls}")

    except Exception as e:
        error_msg = f"❌ Failed to fetch search results: {str(e)}"
        print(error_msg)
        st.error(error_msg)
    return allowed_urls


def check_robots_txt(urls: List[str]) -> List[str]:
    """Checks robots.txt files (simplified)."""
    allowed = []
    for url in urls:
        try:
            # Basic check: Assume allowed if robots.txt fetch/parse fails.
            # In a real app, use RobotFileParser more carefully.
            allowed.append(url)
        except Exception:
            allowed.append(url)
    return allowed # Simplified: Assume allowed for this version


async def crawl_webpages_simple(urls: List[str], query: str) -> List[Dict]:
    """Asynchronously crawls webpages and returns list of {'url': url, 'markdown': markdown}."""
    if not CRAWL4AI_AVAILABLE or not urls:
        return []

    # Simpler config for broader content capture
    md_generator = DefaultMarkdownGenerator() # No aggressive filtering for now
    crawler_config = CrawlerRunConfig(
        markdown_generator=md_generator,
        excluded_tags=["script", "style", "nav", "footer", "aside"],
        only_text=False, # Get more structure initially
        cache_mode=CacheMode.NORMAL, # Use cache for faster re-runs
        user_agent="Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)",
        page_timeout=20000,
        wait_for_network_idle=True,
        network_idle_timeout=3000,
    )
    browser_config = BrowserConfig(headless=True, text_mode=False, light_mode=True)

    results_list = []
    print(f"Starting crawl for {len(urls)} URLs...")
    try:
        async with AsyncWebCrawler(config=browser_config) as crawler:
            crawl_results = await crawler.arun_many(urls, config=crawler_config)
            for res in crawl_results:
                 # Use raw_markdown which might be less clean but includes more
                 markdown_content = res.markdown_v2.raw_markdown if (res and res.markdown_v2 and res.markdown_v2.raw_markdown) else ""
                 if markdown_content.strip():
                     results_list.append({'url': res.url, 'markdown': markdown_content.strip()})
    except Exception as e:
        st.error(f"An error occurred during web crawling: {e}")
        print(f"Crawling error: {e}")

    print(f"Crawling finished. Got {len(results_list)} valid markdown results.")
    return results_list


# --- Resource Processing Workflow ---
async def find_and_synthesize_resources(user_id: str, emotion_id: str):
    """Full workflow: Search, Crawl, Synthesize, Save."""
    st.info("Starting resource discovery...")
    emotion_entry = get_emotion_entry(user_id, emotion_id)
    if not emotion_entry or 'analysis' not in emotion_entry:
        st.error("Cannot find resources: Emotion analysis data is missing.")
        return

    analysis = emotion_entry['analysis']
    primary_emotion = analysis.get('primary_emotion', 'emotional challenge')
    triggers = analysis.get('triggers', [])

    # 1. Search
    with st.spinner("Searching the web for relevant pages..."):
        search_term = f"{primary_emotion} coping strategies {' '.join(triggers)}"
        urls = get_web_urls(search_term, num_results=3) # Limit number of pages

    if not urls:
        st.warning("Could not find suitable web pages for your topic.")
        return

    # 2. Crawl
    with st.spinner(f"Attempting to crawl {len(urls)} web pages..."):
        # Run the async crawl function
        crawled_content_list = await crawl_webpages_simple(urls, search_term)

    if not crawled_content_list:
        st.warning("Found pages, but failed to extract useful content.")
        return

    # 3. Synthesize
    with st.spinner("Synthesizing information from crawled pages using AI..."):
        growth_plan = get_growth_plan(user_id, emotion_id) # Get plan for context
        synthesis_result = synthesize_resources(analysis, growth_plan, crawled_content_list)

    # 4. Save & Update UI
    if isinstance(synthesis_result, dict) and "error" not in synthesis_result and "raw_response" not in synthesis_result:
        save_synthesized_resource(user_id, emotion_id, synthesis_result)
        # Update points
        user_data = get_user_data(user_id)
        if user_data:
             user_data['points'] = user_data.get('points', 0) + 15
             save_user_data(user_id, user_data)
        st.success("Successfully found and synthesized resources!")
        # Set state to view the newly created resource
        st.session_state.selected_resource_emotion_id = emotion_id
        st.session_state.current_view = "view_resource" # Navigate to view resource
        st.rerun()
    else:
        st.error("Failed to synthesize the gathered information.")
        st.json(synthesis_result) # Show error or raw response


# --- UI Components ---

def render_login_page():
    st.title("🌱 Emotion to Action")
    st.subheader("Transform Emotional Experiences into Personal Growth")
    st.markdown("Welcome! Please log in or sign up.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Login")
        login_username = st.text_input("Username", key="login_username").strip()
        login_password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login"):
            user_data = st.session_state.user_db.get(login_username)
            # IMPORTANT: Use password hashing in a real app!
            if user_data and user_data.get('password') == login_password:
                st.session_state.authenticated = True
                st.session_state.current_user = login_username
                st.session_state.current_view = "main"
                st.rerun()
            else:
                st.error("Invalid username or password.")
    with col2:
        st.subheader("Sign Up")
        signup_username = st.text_input("Choose Username", key="signup_username").strip()
        signup_password = st.text_input("Choose Password", type="password", key="signup_password")
        signup_confirm = st.text_input("Confirm Password", type="password", key="signup_confirm")
        if st.button("Sign Up"):
            if not signup_username or not signup_password:
                 st.error("Username and password cannot be empty.")
            elif signup_username in st.session_state.user_db:
                st.error("Username already taken.")
            elif signup_password != signup_confirm:
                st.error("Passwords do not match.")
            else:
                # IMPORTANT: Use password hashing in a real app!
                st.session_state.user_db[signup_username] = {
                    'password': signup_password,
                    'joined_date': datetime.datetime.now().isoformat(),
                    'premium': False, 'streak': 0, 'points': 0, 'goals': {}
                }
                st.success("Account created! You can now login.")

def render_sidebar():
    st.sidebar.title("Navigation")
    user_id = st.session_state.current_user
    user_data = get_user_data(user_id)
    if not user_data: return # Should not happen if authenticated

    st.sidebar.write(f"👋 Hello, {user_id}!")
    st.sidebar.metric("Points", user_data.get('points', 0))
    st.sidebar.metric("Streak", f"{user_data.get('streak', 0)} Days")

    pages = {
        "Dashboard": "main", "Journal": "journal", "Resources": "resources",
        "Community": "community", "Profile": "profile"
    }
    current_page_name = next((name for name, view in pages.items() if st.session_state.current_view == view), "Dashboard")
    selected_page = st.sidebar.radio("Go to:", options=list(pages.keys()), index=list(pages.keys()).index(current_page_name), key="nav_radio")

    # Navigate if selection changed
    if pages[selected_page] != st.session_state.current_view:
        st.session_state.current_view = pages[selected_page]
        # Reset context when changing main sections
        st.session_state.current_emotion_id = None
        st.session_state.selected_resource_emotion_id = None
        st.rerun()

    st.sidebar.divider()
    if st.sidebar.button("Logout"):
        # Clear all session state keys related to user session
        keys_to_reset = ['authenticated', 'current_user', 'current_view', 'current_emotion_id', 'selected_resource_emotion_id']
        for key in keys_to_reset:
             if key in st.session_state:
                 del st.session_state[key]
        # Re-initialize to default state
        initialize_app_state()
        st.rerun()


def render_main_dashboard():
    st.title("🌱 Your Growth Dashboard")
    user_id = st.session_state.current_user
    st.markdown(f"Welcome back, {user_id}! What would you like to do?")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📝 New Journal Entry", use_container_width=True):
            st.session_state.current_view = "journal"
            st.rerun()
    with col2:
        if st.button("👥 Explore Community", use_container_width=True):
            st.session_state.current_view = "community"
            st.rerun()

    st.divider()
    st.subheader("Recent Emotional Entries")
    emotion_history = get_user_emotion_history(user_id)
    if not emotion_history:
        st.info("Start journaling to track your emotional journey.")
    else:
        for entry in sorted(emotion_history, key=lambda x: x['timestamp'], reverse=True)[:3]:
             analysis = entry.get('analysis', {})
             emotion_id = entry.get('id')
             with st.container(border=True):
                 st.write(f"**{analysis.get('primary_emotion', 'N/A')}** (Intensity: {analysis.get('intensity', '?')}/10)")
                 st.caption(f"Recorded: {entry.get('timestamp', '')[:16]}")
                 st.write(entry.get('journal_entry', '')[:100] + "...")
                 if st.button("View Details", key=f"view_dash_{emotion_id}", type="secondary"):
                     st.session_state.current_emotion_id = emotion_id
                     st.session_state.current_view = "emotion_analysis"
                     st.rerun()

    st.divider()
    st.subheader("Recent Community Posts")
    community_posts = get_community_posts(3)
    if not community_posts:
        st.info("No community posts yet.")
    else:
         for post in community_posts:
             with st.container(border=True):
                 st.write(f"**{post.get('title', 'No Title')}** by {post.get('user_id', 'Unknown')}")
                 st.caption(f"❤️ {post.get('likes', 0)} | 💬 {len(post.get('comments', []))}")
                 st.write(post.get('content', '')[:80] + "...")
                 # Add view button if needed


def render_journal_page():
    st.title("📝 Emotional Journal")
    st.write("Reflect on your feelings and experiences.")

    journal_entry = st.text_area("What are you feeling right now?", height=200, key="journal_input")

    if st.button("Analyze My Emotions", type="primary"):
        if not journal_entry.strip():
            st.warning("Please write something in your journal entry.")
        else:
            with st.spinner("Analyzing your emotions..."):
                analysis = analyze_emotion(journal_entry)
                if "error" in analysis or "raw_response" in analysis:
                    st.error("Failed to analyze emotions.")
                    st.json(analysis)
                else:
                    user_id = st.session_state.current_user
                    emotion_id = save_emotion_entry(user_id, journal_entry, analysis)
                    # Update streak/points
                    user_data = get_user_data(user_id)
                    if user_data:
                        user_data['streak'] = user_data.get('streak', 0) + 1
                        user_data['points'] = user_data.get('points', 0) + 10
                        save_user_data(user_id, user_data)
                    st.success("Analysis complete!")
                    st.session_state.current_emotion_id = emotion_id
                    st.session_state.current_view = "emotion_analysis"
                    st.rerun()

    st.divider()
    st.subheader("Past Entries")
    user_id = st.session_state.current_user
    emotion_history = get_user_emotion_history(user_id)
    if not emotion_history:
        st.info("No past entries yet.")
    else:
        for entry in sorted(emotion_history, key=lambda x: x['timestamp'], reverse=True):
            analysis = entry.get('analysis', {})
            emotion_id = entry.get('id')
            with st.expander(f"{analysis.get('primary_emotion', 'Entry')} - {entry.get('timestamp', '')[:10]}"):
                st.caption("Journal Entry:")
                st.markdown(f"> {entry.get('journal_entry', '')}")
                st.caption("Analysis Summary:")
                st.json(analysis, expanded=False) # Keep summary collapsed
                if st.button("View Full Details", key=f"view_journal_{emotion_id}"):
                    st.session_state.current_emotion_id = emotion_id
                    st.session_state.current_view = "emotion_analysis"
                    st.rerun()


def render_emotion_analysis():
    st.title("🧠 Emotion Analysis")
    user_id = st.session_state.current_user
    emotion_id = st.session_state.current_emotion_id

    if not emotion_id:
        st.warning("No emotion entry selected. Go back to Journal.")
        if st.button("Go to Journal"): st.session_state.current_view = "journal"; st.rerun()
        return

    entry = get_emotion_entry(user_id, emotion_id)
    if not entry:
        st.error("Selected entry not found.")
        st.session_state.current_emotion_id = None # Reset invalid ID
        if st.button("Go to Journal"): st.session_state.current_view = "journal"; st.rerun()
        return

    analysis = entry.get('analysis', {})
    st.subheader("Your Journal Entry")
    st.markdown(f"> {entry.get('journal_entry', '')}")
    st.divider()
    st.subheader("AI Analysis Results")
    if "error" in analysis or "raw_response" in analysis or not analysis:
         st.warning("Analysis data is missing or incomplete.")
         st.json(analysis)
    else:
         # Display analysis fields nicely (using columns, metrics, etc.)
         col1, col2 = st.columns(2)
         with col1:
             st.metric("Primary Emotion", analysis.get('primary_emotion', 'N/A'))
             st.metric("Intensity (1-10)", str(analysis.get('intensity', 'N/A')))
         with col2:
              st.write("**Potential Triggers:**")
              st.write(" ".join([f"`{t}`" for t in analysis.get('triggers', [])]) or "None identified")
              st.write("**Emotional Patterns:**")
              st.write(" ".join([f"`{p}`" for p in analysis.get('patterns', [])]) or "None identified")

         st.write("**Growth Opportunities:**")
         for opp in analysis.get('growth_opportunities', []): st.write(f"- {opp}")
         st.write("**Suggested Action Steps:**")
         for step in analysis.get('action_steps', []): st.write(f"- {step}")

    st.divider()
    st.subheader("Next Steps")
    col1, col2, col3 = st.columns(3)

    # 1. Growth Plan
    with col1:
        plan = get_growth_plan(user_id, emotion_id)
        if plan:
             if st.button("View Growth Plan", use_container_width=True):
                 st.session_state.current_view = "growth_plan"
                 st.rerun()
        else:
            if st.button("💡 Create Growth Plan", type="primary", use_container_width=True):
                st.session_state.current_view = "create_growth_plan"
                st.rerun()

    # 2. Resources
    with col2:
        resource = get_synthesized_resource(user_id, emotion_id)
        if resource:
             if st.button("View Resources", use_container_width=True):
                 st.session_state.selected_resource_emotion_id = emotion_id
                 st.session_state.current_view = "view_resource"
                 st.rerun()
        else:
            # Button to trigger the find_and_synthesize_resources async workflow
            find_button_disabled = not CRAWL4AI_AVAILABLE # Disable if library missing
            if st.button("🔎 Find & Synthesize Resources", type="primary", use_container_width=True, disabled=find_button_disabled):
                 # --- Trigger Async Workflow ---
                 try:
                     # Run the async function using asyncio event loop
                     loop = asyncio.new_event_loop()
                     asyncio.set_event_loop(loop)
                     # Use st.spinner context manager around the async call
                     with st.spinner("Finding and synthesizing resources... This may take a minute or two."):
                         loop.run_until_complete(find_and_synthesize_resources(user_id, emotion_id))
                     loop.close()
                     # Rerun is handled inside the function on success
                 except Exception as e:
                     st.error(f"Error starting resource process: {e}")
                     print(f"Error launching find_and_synthesize: {e}")

    # 3. Back Button
    with col3:
        if st.button("Back to Journal", use_container_width=True):
            st.session_state.current_view = "journal"
            st.session_state.current_emotion_id = None
            st.rerun()

def render_create_growth_plan():
    st.title("💡 Create Your Growth Plan")
    user_id = st.session_state.current_user
    emotion_id = st.session_state.current_emotion_id
    if not emotion_id: st.warning("No entry selected."); return
    entry = get_emotion_entry(user_id, emotion_id)
    if not entry or 'analysis' not in entry: st.error("Analysis data missing."); return

    analysis = entry['analysis']
    st.write(f"Creating plan based on feeling **{analysis.get('primary_emotion')}**.")

    st.subheader("Your Goals (Optional)")
    user_data = get_user_data(user_id)
    current_goals = user_data.get('goals', {}) if user_data else {}
    with st.form("goals_form_plan"):
        goal1 = st.text_input("Goal 1:", value=current_goals.get("goal1", ""), key="gp_goal1")
        goal2 = st.text_input("Goal 2:", value=current_goals.get("goal2", ""), key="gp_goal2")
        submitted = st.form_submit_button("✨ Generate Growth Plan")
        if submitted:
             user_goals = {"goal1": goal1, "goal2": goal2}
             # Save goals back to user profile
             if user_data:
                  user_data['goals'] = {k: v for k, v in user_goals.items() if v}
                  save_user_data(user_id, user_data)

             with st.spinner("Generating your plan..."):
                 plan_data = generate_growth_plan(analysis, user_goals)
                 if "error" in plan_data or "raw_response" in plan_data:
                      st.error("Failed to generate plan.")
                      st.json(plan_data)
                 else:
                      save_growth_plan(user_id, emotion_id, plan_data)
                      # Update points
                      if user_data:
                           user_data['points'] = user_data.get('points', 0) + 20
                           save_user_data(user_id, user_data)
                      st.success("Growth plan generated!")
                      st.session_state.current_view = "growth_plan"
                      st.rerun()

    if st.button("Cancel"): st.session_state.current_view = "emotion_analysis"; st.rerun()

def render_growth_plan():
    st.title("🚀 Your Growth Plan")
    user_id = st.session_state.current_user
    emotion_id = st.session_state.current_emotion_id
    if not emotion_id: st.warning("No entry selected."); return

    plan = get_growth_plan(user_id, emotion_id)
    entry = get_emotion_entry(user_id, emotion_id) # For context

    if not plan: st.error("Plan not found."); return
    if entry: st.caption(f"Relates to entry from {entry.get('timestamp', '')[:10]} ({entry['analysis'].get('primary_emotion')})")

    if "error" in plan or "raw_response" in plan:
         st.warning("Plan data incomplete.")
         st.json(plan)
    else:
         st.subheader("🗓️ Short-term Actions")
         for action in plan.get('short_term_actions', []): st.checkbox(action)
         st.subheader("🧘 Medium-term Practices")
         for practice in plan.get('medium_term_practices', []): st.checkbox(practice)
         st.subheader("🌱 Long-term Changes")
         for change in plan.get('long_term_changes', []): st.checkbox(change)
         st.subheader("❓ Reflection Prompts")
         for prompt in plan.get('reflection_prompts', []): st.write(f"- {prompt}")
         st.subheader("📊 Success Metrics")
         for metric in plan.get('success_metrics', []): st.write(f"- {metric}")

    if st.button("Back to Analysis"): st.session_state.current_view = "emotion_analysis"; st.rerun()


def render_resources():
    """Lists saved synthesized resources."""
    st.title("📚 Synthesized Resources")
    st.write("Review the insights gathered from web resources related to your past journal entries.")

    user_id = st.session_state.current_user
    all_resources = get_all_synthesized_resources(user_id)

    if not all_resources:
        st.info("No resources have been synthesized yet. Use the 'Find & Synthesize Resources' button on an emotion analysis page.")
        return

    # Create a mapping from emotion_id to entry details for display
    emotion_map = {entry['id']: entry for entry in get_user_emotion_history(user_id)}

    st.write(f"Found {len(all_resources)} synthesized resource set(s).")

    for emotion_id, resource_synthesis in all_resources.items():
        entry_context = emotion_map.get(emotion_id)
        title = "Resource Synthesis"
        if entry_context:
             title += f" (Related to: {entry_context['analysis'].get('primary_emotion', 'Entry')} on {entry_context['timestamp'][:10]})"

        with st.expander(title):
             st.json(resource_synthesis, expanded=False) # Show collapsed JSON for now
             if st.button("View Formatted Resource", key=f"view_res_{emotion_id}"):
                 st.session_state.selected_resource_emotion_id = emotion_id
                 st.session_state.current_view = "view_resource"
                 st.rerun()


def render_view_resource():
    """Displays a single formatted synthesized resource."""
    st.title("📄 Synthesized Resource Details")
    user_id = st.session_state.current_user
    emotion_id = st.session_state.selected_resource_emotion_id

    if not emotion_id:
         st.warning("No resource selected to view.")
         if st.button("Go to Resources List"): st.session_state.current_view = "resources"; st.rerun()
         return

    resource_data = get_synthesized_resource(user_id, emotion_id)
    entry_context = get_emotion_entry(user_id, emotion_id)

    if not resource_data:
         st.error("Selected resource data not found.")
         if st.button("Go to Resources List"): st.session_state.current_view = "resources"; st.rerun()
         return

    if entry_context:
         st.caption(f"Relates to entry from {entry_context.get('timestamp', '')[:10]} ({entry_context['analysis'].get('primary_emotion')})")

    if "error" in resource_data or "raw_response" in resource_data:
        st.warning("Resource data seems incomplete or improperly formatted.")
        st.json(resource_data)
    else:
        st.markdown("**Key Insights:**")
        for insight in resource_data.get('key_insights', []): st.write(f"- {insight}")

        st.markdown("**Practical Exercises:**")
        for exercise in resource_data.get('practical_exercises', []): st.write(f"- {exercise}")

        st.markdown("**Recommended Readings/Resources:**")
        for reading in resource_data.get('recommended_readings', []): st.write(f"- {reading}")

        st.markdown("**Expert Advice Summary:**")
        st.write(resource_data.get('expert_advice', 'N/A'))

        st.markdown("**Action Plan Integration:**")
        for step in resource_data.get('action_plan', []): st.write(f"- {step}")

        st.markdown("**Sources:**")
        for url in resource_data.get('source_urls', []): st.write(f"- {url}")


    st.divider()
    if st.button("Back to Resources List"):
        st.session_state.current_view = "resources"
        st.session_state.selected_resource_emotion_id = None # Clear selection
        st.rerun()


def render_community_page():
    st.title("👥 Community Hub")
    user_id = st.session_state.current_user

    # AI Suggestions (Optional)
    # ... (Consider adding simplified version if needed) ...

    st.subheader("Share Your Thoughts")
    with st.form("new_post_form", clear_on_submit=True):
        post_title = st.text_input("Title")
        post_content = st.text_area("Message")
        submitted = st.form_submit_button("Post")
        if submitted:
            if post_title and post_content:
                save_community_post(user_id, post_title, post_content)
                # Update points
                user_data = get_user_data(user_id); user_data['points']+=5; save_user_data(user_id, user_data)
                st.success("Posted!")
                # Don't rerun immediately, let success message show
            else:
                st.warning("Title and message are required.")

    st.divider()
    st.subheader("Recent Posts")
    posts = get_community_posts()
    if not posts: st.info("No posts yet."); return

    for post in posts:
        post_id = post['id']
        with st.container(border=True):
            st.markdown(f"#### {post.get('title', 'No Title')}")
            st.caption(f"By: {post.get('user_id', 'Unknown')} on {post.get('timestamp', '')[:16]}")
            st.write(post.get('content', ''))

            # Likes and Comments
            col1, col2 = st.columns([1, 5])
            with col1:
                if st.button(f"❤️ ({post.get('likes', 0)})", key=f"like_{post_id}"):
                    like_post(post_id); st.rerun() # Simple like increment

            with st.expander(f"💬 Comments ({len(post.get('comments', []))})"):
                for comment in sorted(post.get('comments', []), key=lambda c: c['timestamp']):
                    st.markdown(f"**{comment['user_id']}** ({comment['timestamp'][:16]}): {comment['comment']}")
                comment_text = st.text_input("Add comment", key=f"cmt_{post_id}")
                if st.button("Submit", key=f"cmt_btn_{post_id}"):
                    if comment_text:
                        add_comment_to_post(post_id, user_id, comment_text)
                        # Update points
                        user_data = get_user_data(user_id); user_data['points']+=2; save_user_data(user_id, user_data)
                        st.rerun()
                    else: st.warning("Comment cannot be empty.")

def render_profile_page():
    st.title("👤 Your Profile")
    user_id = st.session_state.current_user
    user_data = get_user_data(user_id)
    if not user_data: st.error("User data not found."); return

    st.metric("Username", user_id)
    st.metric("Growth Points", user_data.get('points', 0))
    st.metric("Journaling Streak", user_data.get('streak', 0))
    st.caption(f"Member since: {user_data.get('joined_date', '')[:10]}")

    st.divider()
    st.subheader("Your Growth Goals")
    current_goals = user_data.get('goals', {})
    with st.form("goals_form_profile"):
        goal1 = st.text_input("Goal 1:", value=current_goals.get("goal1", ""), key="prof_goal1")
        goal2 = st.text_input("Goal 2:", value=current_goals.get("goal2", ""), key="prof_goal2")
        submitted = st.form_submit_button("Save Goals")
        if submitted:
            new_goals = {"goal1": goal1.strip(), "goal2": goal2.strip()}
            user_data['goals'] = {k: v for k, v in new_goals.items() if v}
            save_user_data(user_id, user_data)
            st.success("Goals updated!")

    st.divider()
    st.subheader("Settings (Placeholders)")
    st.button("Change Password", disabled=True)
    st.button("Delete Account", disabled=True, type="secondary")

# --- Main Application Logic ---
def main():
    st.set_page_config(layout="wide", page_title="EmotionToAction")

    # Initialize state variables on first run
    initialize_app_state()

    # --- Routing ---
    if not st.session_state.authenticated:
        st.session_state.current_view = "login" # Ensure login view if not auth
        render_login_page()
    else:
        render_sidebar() # Show sidebar only when authenticated

        # Select and render the current view
        view = st.session_state.current_view
        print(f"Rendering view: {view}") # Debug print

        if view == "main": render_main_dashboard()
        elif view == "journal": render_journal_page()
        elif view == "emotion_analysis": render_emotion_analysis()
        elif view == "create_growth_plan": render_create_growth_plan()
        elif view == "growth_plan": render_growth_plan()
        elif view == "resources": render_resources()
        elif view == "view_resource": render_view_resource()
        elif view == "community": render_community_page()
        elif view == "profile": render_profile_page()
        else: # Fallback to dashboard
            st.warning(f"Unknown view '{view}', loading Dashboard.")
            st.session_state.current_view = "main"
            render_main_dashboard()

if __name__ == "__main__":
    main()
