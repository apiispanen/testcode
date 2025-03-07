import streamlit as st
import converted_script
import json
from sklearn.cluster import OPTICS
import numpy as np
from sklearn.cluster import OPTICS
st.html("""
    <style>


    .stLogo {
        height:6rem;
        text-align:center;
        max-width: 46rem;
    }
        
        
    </style>
    """)

from dotenv import load_dotenv
load_dotenv(".env")
import os
st.logo("media/kami_logo.png")

# Load your transformed dataset
with open('data/codebase_chunks.json', 'r') as f:
    transformed_dataset = json.load(f)

# Initialize the VectorDB
base_db = converted_script.VectorDB("base_db")

# Load and process the data
base_db.load_data(transformed_dataset)

# Streamlit app
st.title("OPTICS Clustering and Labeling")

# Sidebar for user inputs
st.sidebar.header("Clustering Parameters")
min_samples = st.sidebar.slider("Minimum Samples", 1, 100, 5)
max_eps = st.sidebar.slider("Maximum Epsilon", 0.1, 10.0, 2.0)

# Perform OPTICS clustering
st.write("Performing OPTICS clustering...")
data = [embedding for embedding in base_db.embeddings]
clusterer = OPTICS(min_samples=min_samples, max_eps=max_eps)
labels = clusterer.fit_predict(data)

# Display clustering results
st.write(f"Number of clusters found: {len(set(labels)) - (1 if -1 in labels else 0)}")
with st.expander("Cluster Labels"):
    st.write(labels)
if st.button("Run Evaluation and Visualization"):
    with st.expander("Evaluation of the database"):
        st.write("We are evaluating the database to see how well it performs.")
        with st.spinner("Evaluating the database..."):
            results5 = converted_script.evaluate_db(base_db, 'data/evaluation_set.jsonl', 5)
            st.write("Evaluation Results (Top 5):", results5)

    # Contextual information
    DOCUMENT_CONTEXT_PROMPT = """
    <document>
    {doc_content}
    </document>
    """

    CHUNK_CONTEXT_PROMPT = """
    Here is the chunk we want to situate within the whole document
    <chunk>
    {chunk_content}
    </chunk>

    Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
    Answer only with the succinct context and nothing else.
    """

    with st.expander("Situating the context"):
        st.write("We are situating the context to improve search retrieval of the chunk.")
        with st.spinner("Situating the context..."):
            jsonl_data = converted_script.load_jsonl('data/evaluation_set.jsonl')
            doc_content = jsonl_data[0]['golden_documents'][0]['content']
            chunk_content = jsonl_data[0]['golden_chunks'][0]['content']

            response = converted_script.situate_context(doc_content, chunk_content, DOCUMENT_CONTEXT_PROMPT, CHUNK_CONTEXT_PROMPT)
            st.write(f"Situated context: {response.content[0].text}")

            # Print cache performance metrics
            st.write(f"Input tokens: {response.usage.input_tokens}")
            st.write(f"Output tokens: {response.usage.output_tokens}")
            st.write(f"Cache creation input tokens: {response.usage.cache_creation_input_tokens}")
            st.write(f"Cache read input tokens: {response.usage.cache_read_input_tokens}")

            # Initialize the ContextualVectorDB
            contextual_db = converted_script.ContextualVectorDB("my_contextual_db")

            # Load and process the data
            contextual_db.load_data(transformed_dataset, parallel_threads=5)


    with st.spinner("Evaluating the contextual database..."):
        with st.expander("Evaluation of the contextual database"):
            st.write("We are evaluating the contextual database to see how well it performs.")
            r5 = converted_script.evaluate_db(contextual_db, 'data/evaluation_set.jsonl', 5)
            st.write("Contextual DB Evaluation Results (Top 5):", r5)

            results5_advanced = converted_script.evaluate_db_advanced(contextual_db, 'data/evaluation_set.jsonl', 5)
            st.write("Advanced Evaluation Results (Top 5):", results5_advanced)

    import matplotlib.pyplot as plt

    # Visualize the clusters
    st.write("We are visualizing the clusters formed by the OPTICS algorithm.")
    with st.spinner("Visualizing the clusters..."):
        # Convert labels to numpy array for easier manipulation
        labels = np.array(labels)

        # Create a scatter plot
        fig, ax = plt.subplots()
        unique_labels = set(labels)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = (labels == k)

            xy = np.array(data)[class_member_mask]
            ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                    markeredgecolor='k', markersize=6)

        ax.set_title('OPTICS Clustering')
        st.pyplot(fig)