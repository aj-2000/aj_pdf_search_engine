import os
import glob
import pickle
import PyPDF2
import nltk
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


# Function to extract text from a PDF file
def extract_text_from_pdf(file_path):
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page in reader.pages:
                text += page.extract_text()
            return text
    except Exception as e:
        print(f"Error extracting text from {file_path}: {str(e)}")
        return ''

# Function to create a synonyms dictionary for all unique words in PDFs
def create_synonyms_dictionary(directory):
    synonyms_dict = {}

    files = glob.glob(os.path.join(directory, '*.pdf'))

    for file_path in files:
        text = extract_text_from_pdf(file_path)
        tokens = word_tokenize(text.lower())
        unique_words = set(tokens)

        for word in unique_words:
            synonyms = set()
            for synset in wordnet.synsets(word):
                for lemma in synset.lemmas():
                    synonym = lemma.name().replace('_', ' ')
                    if synonym != word:
                        synonyms.add(synonym)

            if word in synonyms_dict:
                synonyms_dict[word].extend(list(synonyms))
            else:
                synonyms_dict[word] = list(synonyms)

    return synonyms_dict


# Function to preprocess and tokenize text using NLTK
def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text.lower())

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]

    # Return the preprocessed tokenized text
    return stemmed_tokens


# Function for query expansion
def expand_query(query, synonyms_dict):
    expanded_query = [query]
    if query in synonyms_dict:
        expanded_query.extend(synonyms_dict[query])
    return ' '.join(expanded_query)


def build_tfidf_index(directory, synonyms_dictionary):
    try:
        files = glob.glob(os.path.join(directory, '*.pdf'))
        file_urls = []
        titles = []
        texts = []

        for file_path in files:
            file_url = 'file://' + file_path
            text = extract_text_from_pdf(file_path)
            preprocessed_text = preprocess_text(text)

            file_urls.append(file_url)
            titles.append(os.path.basename(file_path))
            texts.append(' '.join(preprocessed_text))

        # Create a TF-IDF vectorizer
        vectorizer = TfidfVectorizer()

        # Fit the vectorizer on the preprocessed texts
        tfidf_matrix = vectorizer.fit_transform(texts)

        # Calculate the relevance scores using dot product
        relevance_scores = tfidf_matrix.dot(tfidf_matrix.T)

        # Prepare the index
        index = {
            'file_urls': file_urls,
            'titles': titles,
            'relevance_scores': relevance_scores,
            'vectorizer': vectorizer
        }

        return index

    except Exception as e:
        print("Error building TF-IDF index:", str(e))
        return None


# Save the index and synonyms dictionary to files
def save_index_and_synonyms(index, synonyms):
    print(index)
    print(type(index), type(synonyms))
    with open('index.pkl', 'wb') as index_file:
        pickle.dump(index, index_file)

    with open('synonyms.pkl', 'wb') as synonyms_file:
        pickle.dump(synonyms, synonyms_file)


# Load the index and synonyms dictionary from files
def load_index_and_synonyms():
    with open('index.pkl', 'rb') as index_file:
        index = pickle.load(index_file)

    with open('synonyms.pkl', 'rb') as synonyms_file:
        synonyms = pickle.load(synonyms_file)

    return index, synonyms


# Create the Flask app
app = Flask(__name__)


@app.route('/query', methods=['GET'])
def query_index():
    query = request.args.get('query')
    print(query)
    index, synonyms = load_index_and_synonyms()

    # Preprocess the query
    preprocessed_query = ' '.join(preprocess_text(query))

    # Expand the query
    expanded_query = expand_query(preprocessed_query, synonyms)

    # Transform the expanded query using the vectorizer
    query_vector = index['vectorizer'].transform([expanded_query])

    # Calculate the relevance scores using dot product
    relevance_scores = query_vector.dot(index['relevance_scores'])

    # Sort the results by relevance score
    sorted_indices = relevance_scores.A[0].argsort()[::-1]
    sorted_scores = relevance_scores.A[0][sorted_indices]
    sorted_file_urls = [index['file_urls'][idx] for idx in sorted_indices]
    sorted_titles = [index['titles'][idx] for idx in sorted_indices]

    # Prepare the results
    results = []
    for file_url, title, score in zip(sorted_file_urls, sorted_titles, sorted_scores):
        result = {
            "File URL": file_url,
            "Title": title,
            "Relevance Score": score
        }
        results.append(result)

    return jsonify(results)


if __name__ == '__main__':
    # Provide the directory path containing the PDF files
    directory_path = 'docs'
    # Build the TF-IDF index and save it
    synonyms_dictionary = create_synonyms_dictionary(directory_path)
    index = build_tfidf_index(directory_path, synonyms_dictionary)
    save_index_and_synonyms(index, synonyms_dictionary)

    # Run the Flask app
    app.run()

