import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud, STOPWORDS
import networkx as nx
from collections import Counter
import re
from typing import Dict, List, Union, Tuple, Optional, Any
import pandas as pd
from urllib.parse import urlparse
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import io
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('data_viz')

# Initialize NLTK resources
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)

def extract_text_from_data(data: Union[Dict, List[Dict], pd.DataFrame]) -> str:
    """
    Extract text content from various data formats.
    
    Args:
        data: Input data (can be dictionary, list, or DataFrame)
        
    Returns:
        str: Extracted text content
    """
    # Handle different data types
    if isinstance(data, dict):
        # Extract content field if it exists
        return data.get('content', '')
        
    elif isinstance(data, list):
        # Combine content from all dictionaries in the list
        contents = []
        for item in data:
            if isinstance(item, dict):
                content = item.get('content', '')
                if content:
                    contents.append(content)
        return ' '.join(contents)
        
    elif isinstance(data, pd.DataFrame):
        # Check if 'content' column exists
        if 'content' in data.columns:
            return ' '.join(data['content'].astype(str).tolist())
        else:
            # Try to find a column that might contain text
            for col in data.columns:
                if data[col].dtype == 'object':
                    return ' '.join(data[col].astype(str).tolist())
    
    # Return empty string if no text content found
    return ''

def extract_links_from_data(data: Union[Dict, List[Dict], pd.DataFrame]) -> List[Tuple[str, str]]:
    """
    Extract links (URL pairs) from various data formats.
    
    Args:
        data: Input data (can be dictionary, list, or DataFrame)
        
    Returns:
        List[Tuple[str, str]]: List of (source, target) URL pairs
    """
    links = []
    
    # Handle different data types
    if isinstance(data, dict):
        # Get source URL and links from a single dictionary
        source_url = data.get('url', '')
        target_urls = data.get('links', [])
        if source_url and target_urls:
            links.extend([(source_url, target) for target in target_urls])
        
    elif isinstance(data, list):
        # Process each dictionary in the list
        for item in data:
            if isinstance(item, dict):
                source_url = item.get('url', '')
                target_urls = item.get('links', [])
                if source_url and target_urls:
                    links.extend([(source_url, target) for target in target_urls])
        
    elif isinstance(data, pd.DataFrame):
        # Check if 'url' and 'links' columns exist
        if 'url' in data.columns and 'links' in data.columns:
            for _, row in data.iterrows():
                source_url = row['url']
                # Links might be stored as a string representation of a list
                try:
                    if isinstance(row['links'], str):
                        import json
                        target_urls = json.loads(row['links'])
                    else:
                        target_urls = row['links']
                    
                    if source_url and target_urls:
                        links.extend([(source_url, target) for target in target_urls])
                except:
                    pass
    
    return links

def preprocess_text_for_visualization(text: str) -> str:
    """
    Preprocess text for visualization by removing stopwords, punctuation, etc.
    
    Args:
        text: Input text
        
    Returns:
        str: Preprocessed text
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs, email addresses, and special characters
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords and short words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
    
    # Combine back into a string
    return ' '.join(tokens)

def create_word_cloud(
    data: Union[Dict, List[Dict], pd.DataFrame], 
    max_words: int = 200,
    background_color: str = 'white',
    colormap: str = 'viridis'
) -> plt.Figure:
    """
    Create a word cloud visualization from text data.
    
    Args:
        data: Input data
        max_words: Maximum number of words to include
        background_color: Background color for the word cloud
        colormap: Matplotlib colormap name
        
    Returns:
        plt.Figure: Matplotlib figure with the word cloud
    """
    # Extract text from data
    text = extract_text_from_data(data)
    
    # Preprocess text
    text = preprocess_text_for_visualization(text)
    
    if not text:
        # Create empty figure with message if no text available
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No text content available for word cloud generation", 
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig
    
    # Create the word cloud
    wordcloud = WordCloud(
        max_words=max_words,
        background_color=background_color,
        colormap=colormap,
        width=800,
        height=400,
        stopwords=STOPWORDS,
        random_state=42
    ).generate(text)
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    fig.tight_layout()
    
    return fig

def create_frequency_chart(
    data: Union[Dict, List[Dict], pd.DataFrame],
    top_n: int = 20,
    chart_type: str = 'bar',
    ascending: bool = False,
    exclude_common: bool = True
) -> plt.Figure:
    """
    Create a word frequency chart.
    
    Args:
        data: Input data
        top_n: Number of top words to display
        chart_type: Type of chart ('bar', 'horizontal_bar', or 'line')
        ascending: Whether to sort in ascending order
        exclude_common: Whether to exclude common words
        
    Returns:
        plt.Figure: Matplotlib figure with the frequency chart
    """
    # Extract text from data
    text = extract_text_from_data(data)
    
    # Preprocess text
    text = preprocess_text_for_visualization(text)
    
    if not text:
        # Create empty figure with message if no text available
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, "No text content available for frequency chart generation", 
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig
    
    # Count word frequencies
    word_counts = Counter(text.split())
    
    # Remove common words if requested
    if exclude_common:
        common_words = set(['the', 'and', 'to', 'of', 'a', 'in', 'that', 'is', 'it', 'for', 
                            'with', 'as', 'was', 'on', 'are', 'be', 'this', 'by', 'an', 'not'])
        word_counts = {word: count for word, count in word_counts.items() 
                      if word not in common_words}
    
    # Get the top N words
    top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=not ascending)[:top_n]
    words, counts = zip(*top_words) if top_words else ([], [])
    
    # Create the appropriate chart type
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if chart_type == 'bar':
        ax.bar(words, counts, color='skyblue')
        ax.set_xlabel('Words')
        ax.set_ylabel('Frequency')
        plt.xticks(rotation=45, ha='right')
        
    elif chart_type == 'horizontal_bar':
        ax.barh(list(reversed(words)), list(reversed(counts)), color='skyblue')
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Words')
        
    elif chart_type == 'line':
        ax.plot(words, counts, marker='o', linestyle='-', color='skyblue')
        ax.set_xlabel('Words')
        ax.set_ylabel('Frequency')
        plt.xticks(rotation=45, ha='right')
    
    ax.set_title(f'Top {len(words)} Word Frequencies')
    fig.tight_layout()
    
    return fig

def create_network_graph(
    data: Union[Dict, List[Dict], pd.DataFrame],
    max_nodes: int = 50,
    layout: str = 'force_directed',
    show_labels: bool = True,
    sizing_method: str = 'degree'
) -> plt.Figure:
    """
    Create a network graph visualization of links.
    
    Args:
        data: Input data
        max_nodes: Maximum number of nodes to display
        layout: Graph layout algorithm ('force_directed', 'circular', or 'random')
        show_labels: Whether to show node labels
        sizing_method: Method for sizing nodes ('degree', 'page_rank', or 'uniform')
        
    Returns:
        plt.Figure: Matplotlib figure with the network graph
    """
    # Extract links from data
    links = extract_links_from_data(data)
    
    if not links:
        # Create empty figure with message if no links available
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, "No link data available for network graph generation", 
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add edges (links) to the graph
    for source, target in links:
        # Extract domains for better readability
        source_domain = urlparse(source).netloc
        target_domain = urlparse(target).netloc
        G.add_edge(source_domain, target_domain)
    
    # Limit the number of nodes if needed
    if len(G.nodes) > max_nodes:
        # Keep nodes with highest degrees
        degrees = dict(G.degree())
        top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
        top_node_names = [node for node, _ in top_nodes]
        G = G.subgraph(top_node_names)
    
    # Determine node sizes based on the selected method
    if sizing_method == 'degree':
        node_sizes = [10 + 20 * G.degree(node) for node in G.nodes]
    elif sizing_method == 'page_rank':
        pr = nx.pagerank(G)
        node_sizes = [1000 * pr[node] for node in G.nodes]
    else:  # uniform
        node_sizes = [300] * len(G.nodes)
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Choose the appropriate layout
    if layout == 'force_directed':
        pos = nx.spring_layout(G, seed=42)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    else:  # random
        pos = nx.random_layout(G, seed=42)
    
    # Draw the network
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='skyblue', alpha=0.8, ax=ax)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, edge_color='gray', ax=ax)
    
    if show_labels and len(G.nodes) <= 30:  # Only show labels if there aren't too many
        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
    
    ax.set_title(f'Network Graph of {len(G.nodes)} Domains and {len(G.edges)} Links')
    ax.axis('off')
    fig.tight_layout()
    
    return fig

def visualize_sentiment_distribution(sentiment_data: Dict[str, float]) -> plt.Figure:
    """
    Create a visualization of sentiment distribution.
    
    Args:
        sentiment_data: Dictionary mapping sentiment classes to values
        
    Returns:
        plt.Figure: Matplotlib figure with the sentiment visualization
    """
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract labels and values
    labels = list(sentiment_data.keys())
    values = list(sentiment_data.values())
    
    # Create color map (green for positive, red for negative, gray for neutral)
    colors = ['green' if 'positive' in label.lower() else 
              'red' if 'negative' in label.lower() else 
              'gray' for label in labels]
    
    # Create the bar chart
    ax.bar(labels, values, color=colors)
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Score')
    ax.set_title('Sentiment Distribution')
    plt.xticks(rotation=45, ha='right')
    fig.tight_layout()
    
    return fig

def visualize_entity_distribution(entity_data: Dict[str, List[Tuple[str, int]]]) -> plt.Figure:
    """
    Create a visualization of entity distribution.
    
    Args:
        entity_data: Dictionary mapping entity types to lists of (entity, count) tuples
        
    Returns:
        plt.Figure: Matplotlib figure with the entity distribution visualization
    """
    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Process the data
    entity_types = []
    entity_counts = []
    
    for entity_type, entities in entity_data.items():
        if entities:
            count = sum(count for _, count in entities)
            entity_types.append(entity_type)
            entity_counts.append(count)
    
    if not entity_types:
        ax.text(0.5, 0.5, "No entity data available for visualization", 
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig
    
    # Create the bar chart
    colors = plt.cm.tab10(np.linspace(0, 1, len(entity_types)))
    ax.bar(entity_types, entity_counts, color=colors)
    ax.set_xlabel('Entity Type')
    ax.set_ylabel('Count')
    ax.set_title('Entity Type Distribution')
    plt.xticks(rotation=45, ha='right')
    fig.tight_layout()
    
    return fig

if __name__ == "__main__":
    # Test data for visualization
    test_data = {
        "content": """
        Web crawler application is a tool used to automatically browse and index web pages.
        These applications are important for search engines, data mining, and content analysis.
        The crawler searches through websites, follows links, and collects information.
        Modern web crawlers need to respect robots.txt files and website terms of service.
        """,
        "url": "https://example.com",
        "links": [
            "https://example.com/page1",
            "https://example.com/page2",
            "https://othersite.com/page1"
        ]
    }
    
    # Test each visualization type
    print("Creating word cloud...")
    word_cloud_fig = create_word_cloud(test_data)
    word_cloud_fig.savefig("test_word_cloud.png")
    
    print("Creating frequency chart...")
    freq_chart_fig = create_frequency_chart(test_data)
    freq_chart_fig.savefig("test_frequency_chart.png")
    
    print("Creating network graph...")
    network_fig = create_network_graph(test_data)
    network_fig.savefig("test_network_graph.png")
    
    print("Test visualizations saved as PNG files.")
