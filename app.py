# GUI Interface
import tkinter as tk

# Importing models and components
import joblib

# Data Processing Imports
import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.corpus import wordnet

# Load trained components, use same name to keep track of it easier

lr_clf = joblib.load("logistic_regression_model.pkl") # Model
l_encoder = joblib.load("label_encoder.pkl") # Label Encoder


# ----------------------------------------------------------------------------------------------------------------------------
# DATA PREPROCESSING FUNCTIONS AND IMPORTS
# ----------------------------------------------------------------------------------------------------------------------------

# Cleaning the corpus
def clean_text(text):
    
    # Convert to string and lowercase
    text = str(text).lower()

    # Removal of unnecessary text
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+|\[.*?\]\(.*?\)', '', text) # URL and links
    text = re.sub(r'<.*?>+', '', text) # HTML tags
    text = re.sub(r'@\w+', '', text) # Handles (Text that starts with @)
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text) # Punctuation and special characters
    text = re.sub(r'\n', ' ', text) # newlines
    text = re.sub(r'\w*\d\w*', '', text) # Words containing numbers
    text = re.sub(r'\s+', ' ', text) # Extra spaces
    
    return text.strip()

# ----------------------------------------------------------------------------------------------------------------------------
# Stopwords
nltk.download('stopwords') # Install stopwords, if not installed already

stop_words = stopwords.words('english') 
more_stopwords = {"u", "im", "c"}

stop_words = set(stop_words)
stop_words.update(more_stopwords)

# Preserve negation words
stop_words = stop_words - {
    "no",
    "not",
    "nor",
    "never"
}

# Method to remove stopwords
def remove_stopwords(text):
    text = ' '.join(word for word in text.split(' ') if word not in stop_words)
    return text

# ----------------------------------------------------------------------------------------------------------------------------
# Stemming

stemmer = nltk.SnowballStemmer("english")

def stemm_text(text):
    text = ' '.join(stemmer.stem(word) for word in text.split(' '))
    return text

# ----------------------------------------------------------------------------------------------------------------------------
# Full Preprocessing
def preprocess_data(text):

    text = clean_text(text)
    text = remove_stopwords(text)
    text = stemm_text(text)

    return text

# ----------------------------------------------------------------------------------------------------------------------------
# USER INTERFACE DESIGN BELOW
# ----------------------------------------------------------------------------------------------------------------------------

# Main interface
window = tk.Tk()
window.title("Disturbed Statement Detection") # Title
window.geometry("1000x400") # Size

# ----------------------------------------------------------------------------------------------------------------------------
# Center the window
window.update_idletasks()

# Find width and height
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()

window_width = window.winfo_width()
window_height = window.winfo_height()

# Calculate center position
x_position = (screen_width - window_width) // 2
y_position = (screen_height - window_height) // 2

# Move to position
window.geometry(
    f"{window_width}x{window_height}+{x_position}+{y_position}"
)

# ----------------------------------------------------------------------------------------------------------------------------
# Heading
heading_label = tk.Label(
    window,
    text="Disturbed Statement Detection",
    font=("Times New Roman", 18, "bold")
)
heading_label.pack(pady=(30, 10))


# Instructions
instruction_label = tk.Label(
    window,
    text="Enter a statement below:",
    font=("Times New Roman", 11)
)
instruction_label.pack(pady=(0, 5))


# Text Box
statement_textbox = tk.Text(
    window,
    width=60,
    height=8,
    font=("Times New Roman", 11),
    wrap="word"
)
statement_textbox.pack(padx=30, pady=10)

# Store classified output
result_variable = tk.StringVar()
result_variable.set("The classification will appear here.")

# Displaying classification result
result_label = tk.Label(
    window,
    textvariable=result_variable,
    font=("Times New Roman", 12, "bold"),
    wraplength=700
)
result_label.pack(pady=(5, 10))

# ----------------------------------------------------------------------------------------------------------------------------

# Function to read user input
def analyze_statement():

    # User input
    statement = statement_textbox.get("1.0", tk.END).strip()

    if statement == "":
        print("No statement was entered") # Debug
        result_variable.set("Please enter a statement first.")
        return

    # Appply preprocessing
    processed_statement = preprocess_data(statement)

    # Tuned pipeline
    prediction_encoded = lr_clf.predict(
    [processed_statement]
)

    # Decode using the encoder, this shows labels
    prediction_label = l_encoder.inverse_transform(
        prediction_encoded.astype(int)
    )

    # Display result on window
    result_variable.set(f"Mental State: {prediction_label[0]}")

    print("Original statement:", statement) # Debug
    print("Processed statement:", processed_statement) # Debug
    print("Prediction:", prediction_label[0]) # Debug

# Analyze
analyze_button = tk.Button(
    window,
    text="Analyze",
    font=("Times New Roman", 11, "bold"),
    command=analyze_statement
)
analyze_button.pack(pady=10)

# ----------------------------------------------------------------------------------------------------------------------------
# Run app
window.mainloop()