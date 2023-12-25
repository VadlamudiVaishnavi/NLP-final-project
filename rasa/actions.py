# actions.py
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import pandas as pd
import preprocess  # Ensure preprocess.py is accessible
import datetime
import nltk
import random  # Import the random module

# Ensure the necessary NLTK resources are downloaded
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

class ActionFindDiscussion(Action):
    def name(self) -> Text:
        return "action_find_discussion"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        user_message = tracker.latest_message['text']

        # Preprocess the text to retain only certain parts of speech
        preprocessed_text = preprocess.clean_social_media_text(user_message)

        # Define paths
        model_path = "/projectnb/cs505ws/students/saisurya/NLP/models/bertopic_model_boston"
        data_path = "/projectnb/cs505ws/students/saisurya/NLP/topic_labeled_data/boston_with_topics_new.csv"

        # Load the embedding model - make sure this is the same model used during training
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Load the BERTopic model, now providing the embedding model
        topic_model = BERTopic.load(model_path, embedding_model=embedding_model)

        # Load your data
        df = pd.read_csv(data_path)

        # Predict the topic of the user's message
        topics, _ = topic_model.transform([preprocessed_text])
        user_topic = topics[0]

        # Filter the dataframe for discussions related to the predicted topic
        related_discussions = df[df['topic_labels'] == user_topic]

        # Additional filtering based on the current date and time
        current_datetime = datetime.datetime.now()
        current_season = get_season(current_datetime.month)
        current_part_of_day = get_part_of_day(current_datetime.hour)

        # Initial filter based on the season and part of day
        filtered_discussions = related_discussions[
            (related_discussions['Season'] == current_season) &
            (related_discussions['Part_of_Day'] == current_part_of_day)
        ]

        # Check if filtered discussions are empty
        if filtered_discussions.empty:
            # As a fallback, consider all discussions regardless of season or part of day
            filtered_discussions = related_discussions

        # Shuffle the filtered discussions to randomize the selection
        filtered_discussions = filtered_discussions.sample(frac=1, random_state=datetime.datetime.now().microsecond).reset_index(drop=True)

        # Select and respond with the top 3 relevant discussions
        if not filtered_discussions.empty:
            top_discussions = filtered_discussions.head(3)['submission_title'].tolist()
            message = "Here are some relevant posts from this subreddit. Go to the following submission titles to know more:\n" + "\n".join(f"- {title}" for title in top_discussions)
            dispatcher.utter_message(text=message)
        else:
            dispatcher.utter_message(text="I couldn't find any related discussions.")

        return []

# Helper functions for season and part of day
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Fall'

def get_part_of_day(hour):
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 21:
        return 'Evening'
    else:
        return 'Night'

