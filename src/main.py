import streamlit as st
import joblib
import pandas as pd
import os
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

DIRNAME = os.path.dirname(__file__)
STATIC_PATH = os.path.join(DIRNAME, "static")
MODEL_SAVE = "my_random_forest.joblib"
MODEL_PATH = os.path.join(STATIC_PATH, MODEL_SAVE)

SEED = 42


# ======= Data handler class =========

class DataHandler:
    """
    Class to load the dataset and train the model.
    """
    DATASET_PATH = 'data/spam_ham_dataset.csv'

    def __init__(self):
        nltk.download('punkt')
        nltk.download('stopwords')

        self._data = pd.read_csv(self.DATASET_PATH)
        self._tfid = TfidfVectorizer()

        self._preprocess()

    # ======= Private methods =========

    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess the text by removing stopwords and stemming the words
        """
        stop_words = set(nltk.corpus.stopwords.words('english'))
        ps = nltk.stem.PorterStemmer()
        tokens = nltk.tokenize.word_tokenize(text.lower())
        tokens = [
            ps.stem(token) for token in tokens
            if token.isalpha() and token not in stop_words
        ]
        return " ".join(tokens)

    def _preprocess(self):
        """
        Preprocess the dataset
        """
        self._data['processed_text'] = self._data['text'].apply(
            self._preprocess_text
        )
        # Train the tfid vectorizer
        self._tfid.fit_transform(self._data['processed_text']).toarray()

    # ======= Public methods =========

    def prepare_data(self, raw_text: str) -> list:
        """
        Prepare the data for training
        """
        return self._tfid.transform([raw_text])


# ======= Model class =========


class EmailSpamDetector:
    def __init__(self):
        self._init_model()

    def _init_model(self):
        self._model = joblib.load(MODEL_PATH)

    def predict(self, email_content_vectorised: list) -> int:
        return self._model.predict(email_content_vectorised)


# ======= Streamlit app =========


@st.cache_data
def data_init():
    with st.spinner("Loading data..."):
        data_provider = DataHandler()
        return data_provider


@st.cache_resource
def load_model():
    with st.spinner("Loading model..."):
        model = EmailSpamDetector()
        return model


def init_header():
    """
    Initialize the header of the web app.
    """
    st.title("Email Spam detection")
    st.header("Project made for EINIS course.")
    st.divider()


def init_email_input():
    """
    Initialize the email input.
    """
    st.subheader("Email input")
    email_input = st.text_area("Enter your email contents:", "")
    return email_input


def main():
    data_provider = data_init()
    model = load_model()

    init_header()
    email_input = init_email_input()

    if st.button("Submit"):
        email_vectorised = data_provider.prepare_data(email_input)
        prediction = model.predict(email_vectorised)
        if prediction[0] == 1:
            st.write("Email is spam.")
        else:
            st.write("Email is not spam.")


if __name__ == "__main__":
    main()
