import streamlit as st
import pandas as pd
import time
import os
import sys

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from crawlers.basic_crawler import crawl_url, extract_links, crawl_multiple_pages
from crawlers.scrapy_crawler import run_scrapy_crawler, ScrapyLinkExtractor
from data_processing.data_filter import filter_text_data, apply_advanced_filters
from data_processing.dataset_creator import create_dataset, export_dataset
from visualization.data_viz import create_word_cloud, create_frequency_chart, create_network_graph
from ai_analysis.text_analyzer import extract_entities, extract_keywords, categorize_text
from ai_analysis.sentiment_analyzer import analyze_sentiment, analyze_emotions
from utils.helpers import validate_url, format_crawl_results
from utils.search_engine import search_duckduckgo

# Set page config
st.set_page_config(
    page_title="Web Crawler & Data Analysis",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables if they don't exist
if 'crawled_data' not in st.session_state:
    st.session_state.crawled_data = None
if 'filtered_data' not in st.session_state:
    st.session_state.filtered_data = None
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
    
# Flag for auto-starting crawl when URL is selected from search results
if 'start_crawling' not in st.session_state:
    st.session_state.start_crawling = False

# Title and description
st.title("🕸️ Web Crawler & Data Analysis Tool")
st.write("""
This application allows you to crawl websites, extract information based on your search criteria,
filter the data, create datasets, visualize the information, and perform AI-powered analysis.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page:",
    ["Web Crawler", "Data Filtering", "Dataset Creation", "Visualization", "AI Analysis"]
)

# Web Crawler Page
if page == "Web Crawler":
    st.header("Web Crawler")
    
    # Initialize session state for active tab if not already set
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = 0
    
    # Add tabs for different ways to select URLs
    tab_titles = ["Direct URL Input", "Search the Web"]
    active_tab = st.session_state.active_tab
    crawler_tabs = st.tabs(tab_titles)
    
    # First tab: Direct URL input (original functionality)
    with crawler_tabs[0]:
        # Initialize URL in session state if not already present
        if 'url_input' not in st.session_state:
            st.session_state.url_input = ""
            
        # Function to handle URL input changes
        def update_url_input():
            # Only update selected URL if it's a valid URL
            if st.session_state.url_input and validate_url(st.session_state.url_input):
                st.session_state.selected_url = st.session_state.url_input
            
        # Create URL input with a clear button
        col1, col2 = st.columns([6, 1])
        with col1:
            url = st.text_input(
                "Enter URL to crawl (include http:// or https://)", 
                value=st.session_state.url_input,
                placeholder="e.g., https://example.com",
                key="url_input",
                on_change=update_url_input
            )
        with col2:
            if st.button("Clear URL"):
                # Use a different approach to reset the URL input
                # to avoid the "cannot be modified after widget is instantiated" error
                if 'url_input' in st.session_state:
                    del st.session_state.url_input
                st.session_state.selected_url = None
                st.rerun()
                
        search_terms = st.text_input("Enter search terms (comma-separated)", "")
    
    # Second tab: Web search functionality
    with crawler_tabs[1]:
        st.subheader("Search the Web")
        st.write("Search the internet and select websites to crawl from the results.")
        
        # Display selected URL indicator if a URL has been selected
        if hasattr(st.session_state, 'selected_url') and st.session_state.selected_url:
            st.success(f"URL selected for crawling: {st.session_state.selected_url}")
            if st.button("Change URL"):
                st.session_state.selected_url = None
                st.rerun()
                
        # Search input
        search_query = st.text_input("Enter search query", "")
        max_results = st.slider("Maximum number of results", 5, 20, 10)
        
        # Search button
        if st.button("Search Web"):
            if search_query:
                with st.spinner("Searching the web..."):
                    try:
                        search_results = search_duckduckgo(search_query, max_results=max_results)
                        
                        if search_results:
                            st.success(f"Found {len(search_results)} results")
                            
                            # Display search results with selection option
                            st.subheader("Search Results")
                            selected_url = None
                            
                            for i, result in enumerate(search_results):
                                with st.container():
                                    # Use a wider column for content, narrower for buttons
                                    st.markdown(f"**{i+1}. [{result['title']}]({result['url']})**")
                                    st.write(result['description'])
                                    st.text(f"URL: {result['url']}")
                                    
                                    # Create a horizontal line of buttons
                                    btn_col1, btn_col2 = st.columns(2)
                                    
                                    # Define a callback for when a button is clicked
                                    def select_url_callback(url_to_select, idx):
                                        # Make sure URL has http:// or https:// prefix
                                        if not (url_to_select.startswith('http://') or url_to_select.startswith('https://')):
                                            url_to_select = 'https://' + url_to_select
                                        
                                        # Validate URL
                                        if validate_url(url_to_select):
                                            # Set in session state
                                            st.session_state.selected_url = url_to_select
                                            # Also update the url variable directly
                                            st.session_state.url_input = url_to_select
                                            return True
                                        return False
                                    
                                    # Create button layout with better styling - two columns
                                    with btn_col1:
                                        if st.button("📌 Select", key=f"select_{i}"):
                                            selected_url = result['url']
                                            if select_url_callback(selected_url, i):
                                                st.success(f"Selected: {selected_url}")
                                                st.rerun()  # Force rerun to update UI
                                            else:
                                                st.error(f"Invalid URL format: {selected_url}")
                                    
                                    with btn_col2:            
                                        if st.button("🔍 Crawl", key=f"crawl_{i}"):
                                            selected_url = result['url']
                                            if select_url_callback(selected_url, i):
                                                # Set a flag to start crawling immediately
                                                st.session_state.start_crawling = True
                                                st.session_state.active_tab = 1  # Stay on search tab
                                                st.rerun()  # Force rerun to update UI
                                            else:
                                                st.error(f"Invalid URL format: {selected_url}")
                                    st.divider()
                        else:
                            st.warning("No search results found. Try a different query.")
                    except Exception as e:
                        st.error(f"Error searching the web: {str(e)}")
            else:
                st.warning("Please enter a search query")
    
    # This line is removed as we're handling the URL directly in the input field
        
    # Crawler options (apply to both direct input and search)
    crawler_type = st.radio("Select crawler type:", ["Basic Crawler", "Scrapy Crawler"])
    
    crawler_options = st.expander("Crawler Options")
    with crawler_options:
        depth = st.slider("Crawl depth (max links to follow from initial page)", 1, 5, 1)
        max_pages = st.slider("Maximum pages to crawl", 1, 50, 5)
        timeout = st.slider("Request timeout (seconds)", 5, 30, 10)
        respect_robots = st.checkbox("Respect robots.txt", value=True)
        
    # Add a direct URL input option for the search tab
    if active_tab == 1:
        # For search tab, add additional URL input if needed
        direct_url_container = st.container()
        with direct_url_container:
            st.subheader("Or Enter a URL Directly")
            direct_url = st.text_input("Enter URL to crawl (include http:// or https://)", 
                                      placeholder="e.g., https://example.com", 
                                      key="direct_url_input")
            
            if st.button("Use This URL"):
                if validate_url(direct_url):
                    st.session_state.selected_url = direct_url
                    st.session_state.url_input = direct_url
                    st.success(f"URL set to: {direct_url}")
                    st.rerun()
                else:
                    st.error("Please enter a valid URL including http:// or https://")
    
    # Add a debug section to display current URL value
    url_debug = st.empty()
    
    # If we're in the search tab but have a selected URL, switch to the direct input tab
    if active_tab == 1 and hasattr(st.session_state, 'selected_url') and st.session_state.selected_url:
        # Update the URL variable to use the selected URL from session state
        url = st.session_state.selected_url
        url_debug.info(f"Using selected URL for crawling: {url}")
    
    # Check if auto-crawling was triggered by the "Crawl Now" button
    auto_crawl = False
    if st.session_state.start_crawling and hasattr(st.session_state, 'selected_url') and st.session_state.selected_url:
        auto_crawl = True
        url = st.session_state.selected_url
        url_debug.info(f"Auto-crawling URL: {url}")
        # Reset the flag
        st.session_state.start_crawling = False
    
    # Only show the button if not auto-crawling
    start_crawl_button = st.button("Start Crawling")
    
    if start_crawl_button or auto_crawl:
        # Re-check the URL from session state
        if hasattr(st.session_state, 'selected_url') and st.session_state.selected_url:
            url = st.session_state.selected_url
        elif active_tab == 1 and 'direct_url_input' in st.session_state and validate_url(st.session_state.direct_url_input):
            # If we're in the search tab and have a valid direct URL, use that
            url = st.session_state.direct_url_input
            
        if not url or not validate_url(url):
            st.error("Please enter a valid URL including http:// or https://")
            url_debug.warning(f"Current URL value: '{url}' - Is it valid? {validate_url(url) if url else False}")
        else:
            # Display progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Process search terms
            terms = [term.strip() for term in search_terms.split(",")] if search_terms else []
            
            try:
                status_text.text("Setting up crawler...")
                progress_bar.progress(10)
                
                # Choose the appropriate crawler
                if crawler_type == "Basic Crawler":
                    status_text.text(f"Crawling {url} with Basic Crawler...")
                    progress_bar.progress(20)
                    
                    if depth > 1:
                        results = crawl_multiple_pages(url, terms, max_pages=max_pages, depth=depth, timeout=timeout)
                    else:
                        content = crawl_url(url, timeout=timeout)
                        links = extract_links(url, content)
                        results = {
                            "url": url,
                            "content": content,
                            "links": links,
                            "search_results": filter_text_data(content, terms) if terms else []
                        }
                else:  # Scrapy Crawler
                    status_text.text(f"Crawling {url} with Scrapy Crawler...")
                    progress_bar.progress(20)
                    results = run_scrapy_crawler(
                        url, 
                        search_terms=terms,
                        depth=depth,
                        max_pages=max_pages,
                        respect_robots=respect_robots
                    )
                
                # Update progress
                progress_bar.progress(90)
                status_text.text("Processing results...")
                
                # Store results in session state
                st.session_state.crawled_data = results
                
                # Final progress update
                progress_bar.progress(100)
                status_text.text("Crawling completed!")
                
                # Display results
                st.subheader("Crawl Results")
                formatted_results = format_crawl_results(results)
                st.write(formatted_results)
                
                # Display some stats
                st.subheader("Crawl Statistics")
                if isinstance(results, dict):
                    st.write(f"Pages crawled: 1")
                    st.write(f"Links found: {len(results.get('links', []))}")
                    st.write(f"Search term matches: {len(results.get('search_results', []))}")
                else:  # List of results from multi-page crawl
                    st.write(f"Pages crawled: {len(results)}")
                    total_links = sum(len(page.get('links', [])) for page in results)
                    st.write(f"Total links found: {total_links}")
                    total_matches = sum(len(page.get('search_results', [])) for page in results)
                    st.write(f"Total search term matches: {total_matches}")
                
            except Exception as e:
                st.error(f"An error occurred during crawling: {str(e)}")
                progress_bar.empty()
                status_text.empty()
    
    # Display previous results if available
    if st.session_state.crawled_data is not None and not st.button("Clear Results"):
        st.subheader("Previous Crawl Results")
        formatted_results = format_crawl_results(st.session_state.crawled_data)
        st.write(formatted_results)

# Data Filtering Page
elif page == "Data Filtering":
    st.header("Data Filtering")
    
    if st.session_state.crawled_data is None:
        st.warning("Please crawl some data first on the Web Crawler page.")
    else:
        st.subheader("Filter the crawled data")
        
        # Basic filtering options
        filter_expander = st.expander("Basic Filters", expanded=True)
        with filter_expander:
            include_terms = st.text_input("Include terms (comma-separated)", "")
            exclude_terms = st.text_input("Exclude terms (comma-separated)", "")
            case_sensitive = st.checkbox("Case sensitive matching", value=False)
        
        # Advanced filtering options
        advanced_expander = st.expander("Advanced Filters")
        with advanced_expander:
            min_word_count = st.slider("Minimum word count", 0, 1000, 0)
            max_word_count = st.slider("Maximum word count", 0, 5000, 5000)
            date_filter = st.checkbox("Filter by date (if available)")
            if date_filter:
                date_range = st.date_input("Select date range", value=[])
            regex_pattern = st.text_input("Custom regex pattern", "")
        
        if st.button("Apply Filters"):
            try:
                # Convert comma-separated terms to lists
                include_list = [term.strip() for term in include_terms.split(",")] if include_terms else []
                exclude_list = [term.strip() for term in exclude_terms.split(",")] if exclude_terms else []
                
                # Apply basic filters first
                filtered_data = filter_text_data(
                    st.session_state.crawled_data, 
                    include_list, 
                    exclude_list, 
                    case_sensitive
                )
                
                # Apply advanced filters if needed
                advanced_filters = {
                    "min_word_count": min_word_count if min_word_count > 0 else None,
                    "max_word_count": max_word_count if max_word_count < 5000 else None,
                    "regex_pattern": regex_pattern if regex_pattern else None,
                    "date_filter": date_range if date_filter and date_range else None
                }
                
                filtered_data = apply_advanced_filters(filtered_data, advanced_filters)
                
                # Store filtered data
                st.session_state.filtered_data = filtered_data
                
                # Display results
                st.subheader("Filtered Results")
                st.write(filtered_data)
                
                # Display statistics
                st.subheader("Filter Statistics")
                if isinstance(filtered_data, dict):
                    st.write(f"Original items: {len(st.session_state.crawled_data.get('content', '').split())}")
                    st.write(f"Filtered items: {len(filtered_data.get('content', '').split())}")
                else:
                    original_count = sum(len(page.get('content', '').split()) for page in st.session_state.crawled_data)
                    filtered_count = sum(len(page.get('content', '').split()) for page in filtered_data)
                    st.write(f"Original word count: {original_count}")
                    st.write(f"Filtered word count: {filtered_count}")
                    st.write(f"Reduction: {((original_count - filtered_count) / original_count * 100):.2f}%")
                
            except Exception as e:
                st.error(f"An error occurred during filtering: {str(e)}")
        
        # Display previous results if available
        if st.session_state.filtered_data is not None and not st.button("Clear Filtered Results"):
            st.subheader("Previous Filtered Results")
            st.write(st.session_state.filtered_data)

# Dataset Creation Page
elif page == "Dataset Creation":
    st.header("Dataset Creation")
    
    data_to_use = st.radio(
        "Select data source:", 
        ["Crawled Data", "Filtered Data"],
        disabled=(st.session_state.crawled_data is None)
    )
    
    if (data_to_use == "Crawled Data" and st.session_state.crawled_data is None) or \
       (data_to_use == "Filtered Data" and st.session_state.filtered_data is None):
        st.warning(f"Please generate some {data_to_use.lower()} first.")
    else:
        source_data = st.session_state.crawled_data if data_to_use == "Crawled Data" else st.session_state.filtered_data
        
        st.subheader("Configure Dataset")
        
        dataset_options = st.expander("Dataset Options", expanded=True)
        with dataset_options:
            include_metadata = st.checkbox("Include metadata (URL, timestamp, etc.)", value=True)
            include_links = st.checkbox("Include extracted links", value=True)
            text_processing = st.multiselect(
                "Text processing options",
                ["Remove stopwords", "Lemmatize", "Lowercase all", "Remove punctuation", "Remove HTML tags"],
                default=["Remove HTML tags"]
            )
            
        output_options = st.expander("Output Options")
        with output_options:
            output_format = st.selectbox(
                "Select output format",
                ["CSV", "JSON", "Excel", "Pandas DataFrame"]
            )
            file_name = st.text_input("Output file name (without extension)", "crawled_dataset")
        
        if st.button("Create Dataset"):
            try:
                # Process options
                processing_options = {
                    "remove_stopwords": "Remove stopwords" in text_processing,
                    "lemmatize": "Lemmatize" in text_processing,
                    "lowercase": "Lowercase all" in text_processing,
                    "remove_punct": "Remove punctuation" in text_processing,
                    "remove_html": "Remove HTML tags" in text_processing
                }
                
                # Create dataset
                dataset = create_dataset(
                    source_data,
                    include_metadata=include_metadata,
                    include_links=include_links,
                    processing_options=processing_options
                )
                
                # Store dataset in session state
                st.session_state.dataset = dataset
                
                # Display dataset preview
                st.subheader("Dataset Preview")
                if isinstance(dataset, pd.DataFrame):
                    st.dataframe(dataset.head(10))
                else:
                    st.write(dataset)
                
                # Option to export
                if st.button("Export Dataset"):
                    extension = output_format.lower()
                    if extension == "pandas dataframe":
                        extension = "csv"  # Default to CSV for DataFrame
                    
                    output_path = export_dataset(dataset, f"{file_name}.{extension}", output_format)
                    st.success(f"Dataset exported successfully as {output_format}!")
                    
                    # Create download button
                    if extension in ["csv", "json", "xlsx"]:
                        with open(output_path, "rb") as file:
                            st.download_button(
                                label=f"Download {output_format} File",
                                data=file,
                                file_name=f"{file_name}.{extension}",
                                mime=f"application/{extension}"
                            )
                
            except Exception as e:
                st.error(f"An error occurred during dataset creation: {str(e)}")
        
        # Display previous dataset if available
        if st.session_state.dataset is not None and not st.button("Clear Dataset"):
            st.subheader("Previous Dataset")
            if isinstance(st.session_state.dataset, pd.DataFrame):
                st.dataframe(st.session_state.dataset.head(10))
            else:
                st.write(st.session_state.dataset)

# Visualization Page
elif page == "Visualization":
    st.header("Data Visualization")
    
    # Check if we have data to visualize
    if st.session_state.crawled_data is None and st.session_state.filtered_data is None:
        st.warning("Please crawl or filter some data first.")
    else:
        # Select data source
        data_source = st.radio(
            "Select data source for visualization:",
            ["Crawled Data", "Filtered Data", "Dataset"],
            disabled=(st.session_state.crawled_data is None)
        )
        
        # Get the appropriate data
        if data_source == "Crawled Data" and st.session_state.crawled_data is not None:
            data = st.session_state.crawled_data
        elif data_source == "Filtered Data" and st.session_state.filtered_data is not None:
            data = st.session_state.filtered_data
        elif data_source == "Dataset" and st.session_state.dataset is not None:
            data = st.session_state.dataset
        else:
            st.warning(f"No {data_source.lower()} available. Please generate it first.")
            data = None
        
        if data is not None:
            # Select visualization type
            viz_type = st.selectbox(
                "Select visualization type:",
                ["Word Cloud", "Word Frequency Chart", "Link Network Graph"]
            )
            
            # Visualization settings
            settings_expander = st.expander("Visualization Settings", expanded=True)
            with settings_expander:
                if viz_type == "Word Cloud":
                    max_words = st.slider("Maximum number of words", 50, 500, 200)
                    background_color = st.selectbox("Background color", ["white", "black", "gray"])
                    colormap = st.selectbox(
                        "Color scheme", 
                        ["viridis", "plasma", "inferno", "magma", "cividis", "Blues", "Reds", "Greens"]
                    )
                    
                elif viz_type == "Word Frequency Chart":
                    top_n = st.slider("Top N words to display", 5, 50, 20)
                    chart_type = st.selectbox("Chart type", ["Bar Chart", "Horizontal Bar", "Line Chart"])
                    sort_order = st.selectbox("Sort order", ["Descending", "Ascending"])
                    exclude_common = st.checkbox("Exclude common words", value=True)
                    
                elif viz_type == "Link Network Graph":
                    max_nodes = st.slider("Maximum number of nodes", 10, 200, 50)
                    layout = st.selectbox("Graph layout", ["Force-directed", "Circular", "Random"])
                    show_labels = st.checkbox("Show node labels", value=True)
                    node_size_by = st.selectbox(
                        "Node size based on", 
                        ["Degree", "Page Rank", "Uniform"]
                    )
            
            # Create visualization button
            if st.button("Generate Visualization"):
                try:
                    # Progress indicator
                    with st.spinner("Generating visualization..."):
                        if viz_type == "Word Cloud":
                            # Create word cloud
                            fig = create_word_cloud(
                                data, 
                                max_words=max_words,
                                background_color=background_color,
                                colormap=colormap
                            )
                            st.pyplot(fig)
                            
                        elif viz_type == "Word Frequency Chart":
                            # Create frequency chart
                            fig = create_frequency_chart(
                                data,
                                top_n=top_n,
                                chart_type=chart_type.lower().replace(" ", "_"),
                                ascending=(sort_order == "Ascending"),
                                exclude_common=exclude_common
                            )
                            st.pyplot(fig)
                            
                        elif viz_type == "Link Network Graph":
                            # Create network graph
                            fig = create_network_graph(
                                data,
                                max_nodes=max_nodes,
                                layout=layout.lower().replace("-", "_"),
                                show_labels=show_labels,
                                sizing_method=node_size_by.lower().replace(" ", "_")
                            )
                            st.pyplot(fig)
                    
                    # Option to download visualization
                    if st.button("Download Visualization"):
                        # Implementation for downloading would go here
                        # We can save the figure to a BytesIO object and create a download button
                        st.info("Download functionality would be implemented here in a real application.")
                    
                except Exception as e:
                    st.error(f"An error occurred while generating the visualization: {str(e)}")

# AI Analysis Page
elif page == "AI Analysis":
    st.header("AI Analysis")
    
    # Check if we have data to analyze
    if st.session_state.crawled_data is None and st.session_state.filtered_data is None:
        st.warning("Please crawl or filter some data first.")
    else:
        # Select data source
        data_source = st.radio(
            "Select data source for analysis:",
            ["Crawled Data", "Filtered Data", "Dataset"],
            disabled=(st.session_state.crawled_data is None)
        )
        
        # Get the appropriate data
        if data_source == "Crawled Data" and st.session_state.crawled_data is not None:
            data = st.session_state.crawled_data
        elif data_source == "Filtered Data" and st.session_state.filtered_data is not None:
            data = st.session_state.filtered_data
        elif data_source == "Dataset" and st.session_state.dataset is not None:
            data = st.session_state.dataset
        else:
            st.warning(f"No {data_source.lower()} available. Please generate it first.")
            data = None
        
        if data is not None:
            # Select analysis type
            analysis_type = st.selectbox(
                "Select analysis type:",
                ["Named Entity Recognition", "Keyword Extraction", "Text Categorization", 
                 "Sentiment Analysis", "Emotion Analysis"]
            )
            
            # Analysis settings
            settings_expander = st.expander("Analysis Settings", expanded=True)
            with settings_expander:
                if analysis_type == "Named Entity Recognition":
                    entity_types = st.multiselect(
                        "Entity types to extract",
                        ["PERSON", "ORGANIZATION", "LOCATION", "DATE", "MONEY", "GPE", "EVENT", "PRODUCT"],
                        default=["PERSON", "ORGANIZATION", "LOCATION"]
                    )
                    min_confidence = st.slider("Minimum confidence score", 0.0, 1.0, 0.5)
                    
                elif analysis_type == "Keyword Extraction":
                    max_keywords = st.slider("Maximum number of keywords", 5, 50, 20)
                    use_phrases = st.checkbox("Extract multi-word phrases", value=True)
                    
                elif analysis_type == "Text Categorization":
                    taxonomy = st.selectbox(
                        "Categorization taxonomy",
                        ["General Topics", "News Categories", "Academic Subjects", "Custom"]
                    )
                    if taxonomy == "Custom":
                        custom_categories = st.text_area(
                            "Enter custom categories (one per line)",
                            "Technology\nHealth\nFinance\nEducation\nEntertainment"
                        )
                    threshold = st.slider("Categorization threshold", 0.0, 1.0, 0.3)
                    
                elif analysis_type == "Sentiment Analysis":
                    sentiment_model = st.selectbox(
                        "Sentiment analysis model",
                        ["VADER", "TextBlob", "spaCy"]
                    )
                    show_breakdown = st.checkbox("Show detailed breakdown", value=True)
                    
                elif analysis_type == "Emotion Analysis":
                    emotion_set = st.selectbox(
                        "Emotion set",
                        ["Basic (Joy, Sadness, Anger, Fear, Surprise)", 
                         "Extended (+ Disgust, Trust, Anticipation)"]
                    )
                    visualize_emotions = st.checkbox("Visualize emotion distribution", value=True)
            
            # Run analysis button
            if st.button("Run Analysis"):
                try:
                    # Progress indicator
                    with st.spinner(f"Running {analysis_type}..."):
                        if analysis_type == "Named Entity Recognition":
                            # Extract entities
                            entities = extract_entities(
                                data,
                                entity_types=entity_types,
                                min_confidence=min_confidence
                            )
                            st.session_state.analysis_results = entities
                            
                            # Display entities
                            st.subheader("Extracted Entities")
                            for entity_type, entities_list in entities.items():
                                if entities_list:
                                    st.write(f"**{entity_type}**")
                                    for entity, count in entities_list:
                                        st.write(f"- {entity}: {count} occurrences")
                            
                        elif analysis_type == "Keyword Extraction":
                            # Extract keywords
                            keywords = extract_keywords(
                                data,
                                max_keywords=max_keywords,
                                use_phrases=use_phrases
                            )
                            st.session_state.analysis_results = keywords
                            
                            # Display keywords
                            st.subheader("Extracted Keywords")
                            for keyword, score in keywords:
                                st.write(f"- {keyword}: relevance score {score:.4f}")
                            
                        elif analysis_type == "Text Categorization":
                            # Get categories
                            if taxonomy == "Custom" and 'custom_categories' in locals():
                                categories = [cat.strip() for cat in custom_categories.split("\n") if cat.strip()]
                            else:
                                categories = None  # Use default categories for the selected taxonomy
                                
                            # Categorize text
                            categorization = categorize_text(
                                data,
                                taxonomy=taxonomy.lower().replace(" ", "_"),
                                custom_categories=categories,
                                threshold=threshold
                            )
                            st.session_state.analysis_results = categorization
                            
                            # Display categorization
                            st.subheader("Text Categorization Results")
                            for category, score in categorization:
                                st.write(f"- {category}: confidence {score:.4f}")
                            
                        elif analysis_type == "Sentiment Analysis":
                            # Analyze sentiment
                            sentiment = analyze_sentiment(
                                data,
                                model=sentiment_model.lower(),
                                detailed=show_breakdown
                            )
                            st.session_state.analysis_results = sentiment
                            
                            # Display sentiment
                            st.subheader("Sentiment Analysis Results")
                            st.write(f"**Overall Sentiment**: {sentiment['overall']}")
                            st.write(f"**Polarity Score**: {sentiment['polarity']:.4f}")
                            
                            if show_breakdown and 'breakdown' in sentiment:
                                st.write("**Sentiment Breakdown**:")
                                for aspect, value in sentiment['breakdown'].items():
                                    st.write(f"- {aspect}: {value:.4f}")
                            
                            # Visualization of sentiment
                            import matplotlib.pyplot as plt
                            fig, ax = plt.subplots()
                            sentiment_values = [max(-1, min(1, sentiment['polarity']))]
                            labels = ['Polarity']
                            colors = ['green' if sentiment_values[0] > 0 else 'red']
                            ax.bar(labels, sentiment_values, color=colors)
                            ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
                            ax.set_ylim(-1, 1)
                            ax.set_title('Sentiment Polarity')
                            st.pyplot(fig)
                            
                        elif analysis_type == "Emotion Analysis":
                            # Analyze emotions
                            is_extended = "Extended" in emotion_set
                            emotions = analyze_emotions(
                                data,
                                extended=is_extended
                            )
                            st.session_state.analysis_results = emotions
                            
                            # Display emotions
                            st.subheader("Emotion Analysis Results")
                            st.write(f"**Dominant Emotion**: {emotions['dominant']}")
                            
                            # Display all emotions
                            st.write("**Emotion Distribution**:")
                            for emotion, score in emotions['scores'].items():
                                st.write(f"- {emotion}: {score:.4f}")
                            
                            # Visualize emotions if requested
                            if visualize_emotions:
                                import matplotlib.pyplot as plt
                                
                                fig, ax = plt.subplots()
                                emotions_dict = emotions['scores']
                                labels = list(emotions_dict.keys())
                                values = list(emotions_dict.values())
                                
                                # Create color map
                                emotion_colors = {
                                    'Joy': 'gold', 
                                    'Sadness': 'royalblue', 
                                    'Anger': 'firebrick',
                                    'Fear': 'darkviolet', 
                                    'Surprise': 'orange',
                                    'Disgust': 'mediumseagreen', 
                                    'Trust': 'deepskyblue', 
                                    'Anticipation': 'coral'
                                }
                                colors = [emotion_colors.get(emotion, 'gray') for emotion in labels]
                                
                                ax.bar(labels, values, color=colors)
                                ax.set_ylabel('Intensity')
                                ax.set_title('Emotion Distribution')
                                plt.xticks(rotation=45)
                                plt.tight_layout()
                                st.pyplot(fig)
                    
                    # Export analysis results option
                    if st.button("Export Analysis Results"):
                        # Implementation for exporting would go here
                        import json
                        import io
                        
                        # Convert results to JSON
                        json_results = json.dumps(st.session_state.analysis_results, indent=2)
                        
                        # Create download button
                        st.download_button(
                            label="Download Results as JSON",
                            data=json_results,
                            file_name=f"{analysis_type.lower().replace(' ', '_')}_results.json",
                            mime="application/json"
                        )
                    
                except Exception as e:
                    st.error(f"An error occurred during analysis: {str(e)}")

# Add footer
st.markdown("---")
st.markdown("Web Crawler & Data Analysis Tool | Developed with Streamlit")
