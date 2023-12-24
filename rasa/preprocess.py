import re
import nltk

# Download required NLTK resources (if not already done)
nltk.download('punkt')

# Sample Emoticons to Text dictionary
EMOTICONS_EMO = {
    ":)": "happy",
    ":(": "sad"
    # Add more mappings here
}

# Define convert_emojis function
def convert_emojis(text):
    for emot in EMOTICONS_EMO:
        text = re.sub(re.escape(emot), ' '.join(EMOTICONS_EMO[emot].replace(",", "").replace(":", "").split()), text)
    return text

# Define clean_social_media_text function
def clean_social_media_text(text):
    text = convert_emojis(text)  # Convert emojis to text
    text = text.replace('|', ' ')
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+', '', text)
    text = re.sub(r'\#\w+', '', text)
    text = re.sub(r'\&\w+;', '', text)
    text = re.sub(r'[\n\r\t]+', ' ', text)
    text = re.sub(r'\bnan\b', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.strip()
    text = text.lower()

    return text


