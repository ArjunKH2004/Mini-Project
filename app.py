import os
import streamlit as st
from googleapiclient.discovery import build
from textblob import TextBlob
import pickle
from tfIdfInheritVectorizer.feature_extraction.vectorizer import TFIDFVectorizer

# Google API Key (ensure you don't expose it publicly)
API_KEY = 'AIzaSyAlM9YHLtaMRuVVp9nf0yCJTQ-cJkhrgUc'

# Function to retrieve YouTube comments
def get_video_comments(video_id):
    comments = []
    youtube = build('youtube', 'v3', developerKey=API_KEY)
    results = youtube.commentThreads().list(
        part='snippet',
        videoId=video_id,
        textFormat='plainText',
        maxResults=100
    ).execute()

    while results:
        for item in results['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)

        if 'nextPageToken' in results:
            results = youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                textFormat='plainText',
                pageToken=results['nextPageToken'],
                maxResults=100
            ).execute()
        else:
            break

    return comments

# Function to classify comments
def classify_comments(comments):
    tfidfconverter = TFIDFVectorizer(max_features=5)
    tfidfconverter.fit(comments)
    
    with open("yt_ai_classifier_model.sav", "rb") as f:
        model = pickle.load(f)
    
    categorized_comments = {'good': [], 'bad': [], 'neutral': []}

    for comment in comments:
        new_text = [comment]
        new_text_transformed = tfidfconverter.transform(new_text).toarray()
        output = model.predict(new_text_transformed)

        if output == 2:
            categorized_comments['good'].append(comment)
        elif output == 0:
            categorized_comments['bad'].append(comment)
        else:
            categorized_comments['neutral'].append(comment)

    return categorized_comments

# Main Streamlit function
def main():
    st.title("YouTube Comment Classifier")

    video_url = st.text_input("Enter the YouTube video URL:", "")
    
    if st.button("Classify Comments") and video_url:
        video_id = video_url.split('v=')[1].split('&')[0] if 'v=' in video_url else video_url.split('/')[-1]

        with st.spinner("Fetching and classifying comments..."):
            comments = get_video_comments(video_id)
            categorized_comments = classify_comments(comments)
        
        st.subheader("Good Comments:")
        st.write("\n".join(categorized_comments['good']))

        st.subheader("Bad Comments:")
        st.write("\n".join(categorized_comments['bad']))

        st.subheader("Neutral Comments:")
        st.write("\n".join(categorized_comments['neutral']))

if __name__ == "__main__":
    main()
