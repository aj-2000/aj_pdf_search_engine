import os
import glob
import pickle
import argparse
from tqdm import tqdm
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob  # Add TextBlob library
from multiprocessing import Pool, cpu_count

import logging
from pdfminer.high_level import extract_text

from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

from transformers import AutoTokenizer, AutoModel, pipeline
import torch


class ZeroShotClassifier:
    def __init__(self, model_name='facebook/bart-large-mnli'):
        self.classifier = pipeline(
            "zero-shot-classification", model=model_name)

    def classify(self, text, candidate_labels):
        return self.classifier(text, candidate_labels)


class TransformerEmbeddings:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()  # Set model to evaluation mode

    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt",
                                truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()


PERCENTAGE_THRESHOLD = 0.1
TOP_DOCUMENTS = 5


def convert_sentiment_to_label(sentiment_score):
    if sentiment_score > 0.2:
        return "positive"
    elif sentiment_score < -0.2:
        return "negative"
    else:
        return "neutral"


class PDFProcessor:

    """Handles PDF extraction and text preprocessing."""

    @staticmethod
    def extract_text_by_page(file_path):
        """Extracts text content page by page from a PDF file."""
        texts = []
        try:
            full_text = extract_text(file_path).replace("\n", " ")
            texts = [text for text in full_text.split('\f')]
        except Exception as e:
            logging.error(f"Error extracting text from {file_path}: {e}")
        return texts

    @staticmethod
    def classify_text(text, classifier, labels):
        """Classify the text into given labels using zero-shot classification."""
        classification_result = classifier.classify(text, labels)
        return classification_result['labels'][0]  # Return the top label

    @staticmethod
    def preprocess(text):
        """Preprocesses a given text."""
        tokens = [token.lower()
                  for token in word_tokenize(text) if token.isalpha()]
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        lemmatizer = WordNetLemmatizer()
        pos_tags = nltk.pos_tag(tokens)
        tokens = [lemmatizer.lemmatize(token, PDFProcessor._get_wordnet_pos(
            pos_tag)) for token, pos_tag in pos_tags]
        return ' '.join(tokens)

    @staticmethod
    def _get_wordnet_pos(tag):
        """Maps POS tag to first character used by WordNetLemmatizer."""
        tag_dict = {
            "J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV
        }
        return tag_dict.get(tag[0].upper(), wordnet.NOUN)


class Doc2VecProcessor:
    """Handles Doc2Vec related functionalities."""

    @staticmethod
    def train_doc2vec_model(docs, vector_size=35, window=5, min_count=2, workers=4, epochs=50):
        """Train a Doc2Vec model with the provided documents."""
        tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[
                                      str(i)]) for i, _d in enumerate(docs)]
        model = Doc2Vec(vector_size=vector_size, window=window,
                        min_count=min_count, workers=workers)
        model.build_vocab(tagged_data)
        model.train(tagged_data, total_examples=model.corpus_count,
                    epochs=epochs)
        return model

    @staticmethod
    def infer_vector(model, doc):
        """Infer vector for a document using a trained Doc2Vec model."""
        return model.infer_vector(word_tokenize(doc.lower()))


class IndexBuilder:
    """Handles index building operations."""

    def __init__(self, mode="tfidf"):
        self.mode = mode

    def _process_file(self, file_path):
        """Process a single file by extracting and preprocessing its text page by page."""
        pages = PDFProcessor.extract_text_by_page(file_path)
        processed_pages = [PDFProcessor.preprocess(page) for page in pages]

        # Analyze sentiment for each page
        sentiment_scores = [
            TextBlob(page).sentiment.polarity for page in processed_pages]
        return processed_pages, sentiment_scores

    def build(self, directory_path, batch_size=1000, use_transformer=False, use_zero_shot=False, candidate_labels=None):
        """Builds the index from a given directory of PDF files page by page."""
        file_paths = glob.glob(os.path.join(directory_path, '*.pdf'))

        # Create lists to store processed data
        processed_pages = []
        document_pages = []
        sentiment_scores = []

        # Process files in chunks (batches)
        for i in range(0, len(file_paths), batch_size):
            batch_paths = file_paths[i: i + batch_size]
            with Pool(cpu_count()) as pool:
                # Process a batch of files in parallel and extract text by pages
                processed_data = list(
                    tqdm(pool.imap(self._process_file, batch_paths), total=len(batch_paths)))

            for batch_processed_pages, batch_sentiment_scores in processed_data:
                processed_pages.extend(batch_processed_pages)
                sentiment_scores.extend(batch_sentiment_scores)

            document_pages.extend(
                [(fp, idx) for fp, doc in zip(batch_paths, batch_processed_pages) for idx in range(len(doc))])

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(processed_pages)

        if use_zero_shot:
            classifier = ZeroShotClassifier()
            for idx, page in enumerate(processed_pages):
                label = PDFProcessor.classify_text(
                    page, classifier, candidate_labels)
                # Store or use the label as needed, e.g., add to your data dictionary
                data['labels'][idx] = label

        if use_transformer:
            transformer_processor = TransformerEmbeddings()
            data['document_vectors'] = [
                transformer_processor.get_embedding(page) for page in processed_pages]
        else:
            data = {
                'vectorizer': vectorizer,
                'document_pages': document_pages,
                'sentiment_scores': sentiment_scores  # Store sentiment scores in the index
            }

            if self.mode == "lsi":
                lsi_model = TruncatedSVD(n_components=50)
                data['lsi_matrix'] = lsi_model.fit_transform(tfidf_matrix)
                data['lsi_model'] = lsi_model
            elif self.mode == "doc2vec":
                d2v_processor = Doc2VecProcessor()
                self.d2v_model = d2v_processor.train_doc2vec_model(
                    processed_pages)
                data['d2v_model'] = self.d2v_model
                data['document_vectors'] = [self.d2v_model.dv[i]
                                            for i in range(len(processed_pages))]
            else:
                data['tfidf_matrix'] = tfidf_matrix
        return data


class SearchEngine:
    """Handles search functionalities."""

    def __init__(self, index_data, mode):
        self.mode = mode
        self.data = index_data
        if self.mode == 'doc2vec':
            self.d2v_model = index_data['d2v_model']

    def query(self, text, top_k=10):
        """Queries the search engine and retrieves relevant pages."""
        preprocessed_query = PDFProcessor.preprocess(text)

        if self.mode == "transformer":
            transformer_processor = TransformerEmbeddings()
            query_vector = transformer_processor.get_embedding(text)
            scores = cosine_similarity(
                [query_vector], self.data['document_vectors'])  # Make sure 'document_vectors' are also from transformer
            similarities = scores[0]
        elif self.mode == "lsi":
            query_vector = self.data['vectorizer'].transform(
                [preprocessed_query])
            lsi_query_vector = self.data['lsi_model'].transform(query_vector)
            similarities = cosine_similarity(
                self.data['lsi_matrix'], lsi_query_vector).flatten()
        elif self.mode == "doc2vec":
            query_vector = Doc2VecProcessor.infer_vector(
                self.d2v_model, preprocessed_query)
            scores = cosine_similarity(
                [query_vector], self.data['document_vectors'])
            similarities = scores[0]
        else:
            query_vector = self.data['vectorizer'].transform(
                [preprocessed_query])
            similarities = cosine_similarity(
                self.data['tfidf_matrix'], query_vector).flatten()

        top_indices = similarities.argsort()[:-top_k - 1:-1]
        scores = similarities[top_indices]
        paths = [(self.data['document_pages'][index][0],
                  self.data['document_pages'][index][1]) for index in top_indices]

        # Calculate total number of pages for each document
        total_pages_per_doc = {}
        for doc, page in self.data['document_pages']:
            total_pages_per_doc[doc] = total_pages_per_doc.get(doc, 0) + 1

        # Aggregate the similarity scores for each document
        doc_similarity_aggregate = {}
        for index in similarities.argsort()[:-int(PERCENTAGE_THRESHOLD * len(similarities)) - 1:-1]:
            doc_path = self.data['document_pages'][index][0]
            # Aggregate the similarity score for the document
            doc_similarity_aggregate[doc_path] = doc_similarity_aggregate.get(
                doc_path, 0) + similarities[index]

        # Normalize the similarity score aggregate by total number of pages
        normalized_similarity = {}
        for doc, aggregate_score in doc_similarity_aggregate.items():
            normalized_similarity[doc] = aggregate_score / \
                total_pages_per_doc[doc]

        # Sort documents by normalized similarity
        sorted_docs = sorted(normalized_similarity.items(),
                             key=lambda kv: kv[1], reverse=True)

        # Retrieve the labels for the top results
        labels = [self.data['labels'][index] for index in top_indices]

        return paths, scores, labels, sorted_docs[:TOP_DOCUMENTS]


def save_index(index_file, data):
    """Save index data to a file."""
    with open(index_file, 'wb') as f:
        pickle.dump(data, f)


def load_index(index_file):
    """Load index data from a file."""
    with open(index_file, 'rb') as f:
        return pickle.load(f)


def get_multiline_input(prompt, end_keyword="END"):
    """Get multiline input from the user until the end keyword is entered."""
    print(prompt, f"(Type '{end_keyword}' on a new line to finish)")
    lines = []
    while True:
        try:
            line = input()
            if line.strip().upper() == end_keyword:
                break
            lines.append(line)
        except EOFError:  # This handles Ctrl+D
            break
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Build an index and search PDFs.")
    parser.add_argument(
        '--index', type=str, default='index_data.pkl', help='Path to the index file.')
    parser.add_argument('--docs', type=str, default='docs',
                        help='Path to the directory containing PDF documents.')
    parser.add_argument('--update-index', action='store_true',
                        help='Update the index if it already exists.')
    parser.add_argument('--mode', type=str, choices=['tfidf', 'lsi', 'doc2vec', 'transformer'], default='tfidf',
                        help='The indexing and search mode.')
    args = parser.parse_args()

    if args.mode == 'transformer':
        use_transformer = True
    else:
        use_transformer = False

    # Check if index file exists
    if os.path.exists(args.index) and not args.update_index:
        print("Loading existing index...")
        index_data = load_index(args.index)
    else:
        print("Building new index...")
        indexer = IndexBuilder(args.mode)
        candidate_labels = ['label1', 'label2',
                            'label3']  # Define your labels here
        index_data = indexer.build(args.docs, use_transformer=use_transformer,
                                   use_zero_shot=True, candidate_labels=candidate_labels)
        save_index(args.index, index_data)
        print(f"Index saved to {args.index}")

    search_engine = SearchEngine(index_data, args.mode)

    while True:
        query = get_multiline_input("Enter your search query")

        if not query.strip():  # If the user just presses enter without any input
            print("Empty query, please try again or type 'exit' to stop.")
            continue

        if query.strip().lower() == 'exit':
            break

        results, scores, labels, sorted_docs = search_engine.query(query)
    print("Top pages with highest similarity score:")
    for i, ((path, page), label) in enumerate(zip(results, labels)):
        print(
            f"Document: {path}, Page: {page + 1}, Score: {scores[i]:.4f}, Label: {label}, Sentiment: {convert_sentiment_to_label(index_data['sentiment_scores'][i])}")

    print("Top 5 relevant documents:")
    for doc, count in sorted_docs:
        # Assuming label for the first page represents the document
        doc_label = index_data['labels'][index_data['document_pages'].index(
            (doc, 0))]
        print(
            f"Document: {doc}, Cumulative Score: {count}, Label: {doc_label}")


if __name__ == "__main__":
    main()
