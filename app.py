import streamlit as st
from gradio_client import Client
import pandas as pd
import base64
import io

# Streamlit app
st.set_page_config(page_title="YouTube Sentiment Analysis", page_icon="🎥", layout="wide")
st.title("🎥 YouTube Comments Sentiment Analyzer")

st.markdown("Analyze the sentiment of YouTube video comments with beautiful graphs 📈")

# Inputs
url = st.text_input("Enter YouTube Video URL:")
num_of_comments = st.slider("Number of comments to analyze:", 10, 500, step=10, value=100)

# Button
if st.button("Analyze 🎯"):
    with st.spinner('Analyzing comments... Please wait ⏳'):
        try:
            # API call
            client = Client("yuvarajareddy001/youtube_comments_sentiment")
            result = client.predict(
                url=url,
                num_of_comments=num_of_comments,
                api_name="/youtube_sentiment_analysis"
            )

            # Unpack the tuple
            sentiment_summary = result[0]   # str
            plot_data = result[1]            # dict (with type and plot)
            table_data = result[2]           # dict (headers, data, metadata)

            # Display sentiment summary
            st.subheader("🔎 Overall Sentiment Summary")
            st.success(sentiment_summary)

            # Display sentiment chart
            st.subheader("📊 Sentiment Chart")
            if plot_data['type'] == 'matplotlib':
                # Decode the base64-encoded image string
                image_bytes = base64.b64decode(plot_data['plot'].split(",")[1])

                # Read image bytes as a stream
                image_stream = io.BytesIO(image_bytes)

                # Show the image directly in Streamlit
                st.image(image_stream, caption="Sentiment Chart", use_container_width=True)

            else:
                st.warning("Plot type not supported yet.")

            # Display comment sentiment analysis table
            st.subheader("📝 Comment Sentiment Analysis")
            headers = table_data['headers']
            data_rows = table_data['data']

            df = pd.DataFrame(data_rows, columns=headers)
            st.dataframe(df, use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred: {e}")
