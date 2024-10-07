
import os
from googleapiclient.discovery import build
from textblob import TextBlob
import pickle
from tfIdfInheritVectorizer.feature_extraction.vectorizer import TFIDFVectorizer

API_KEY = 'AIzaSyAlM9YHLtaMRuVVp9nf0yCJTQ-cJkhrgUc'
youtube = build('youtube', 'v3', developerKey=API_KEY)
def get_video_comments(video_id):
    comments = []
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

def classify_comments(comments):
    tfidfconverter = TFIDFVectorizer(max_features=5)
    tfidfconverter.fit(comments)  
    with open("yt_ai_classifier_model.sav", "rb") as f:
      model = pickle.load(f)
    categorized_comments = {'good': [], 'bad': [], 'neutral': []}

    for comment in comments:
        new_text = [comment]
        new_text_transformed = tfidfconverter.transform(new_text).toarray()
        output = (model.predict(new_text_transformed))

        if output == 2:
            categorized_comments['good'].append(comment)
        elif output == 0:
            categorized_comments['bad'].append(comment)
        else:
            categorized_comments['neutral'].append(comment)

    return categorized_comments

def main(video_url):

    video_id = video_url.split('v=')[1].split('&')[0] if 'v=' in video_url else video_url.split('/')[-1]
    
    comments = get_video_comments(video_id)
    categorized_comments = classify_comments(comments)
    
    return categorized_comments

video_url = input("Enter the YouTube video URL: ")
categorized_comments = main(video_url)

print("Good Comments:")
print("\n".join(categorized_comments['good']))
print("\nBad Comments:")
print("\n".join(categorized_comments['bad']))
print("\nNeutral Comments:")
print("\n".join(categorized_comments['neutral']))
