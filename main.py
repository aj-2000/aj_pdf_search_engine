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
from multiprocessing import Pool, cpu_count

import logging
from pdfminer.high_level import extract_text



from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

# Download required NLTK datasets
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')


class PDFProcessor:
    """Handles PDF extraction and text preprocessing."""

    def extract_text_by_page(file_path):
        """Extracts text content page by page from a PDF file."""
        texts = []
        try:
            # Using pdfminer's extract_text which extracts text from entire document
            # Splitting by page can be done by processing the PDF in a more granular way
            full_text = extract_text(file_path).replace("\n", " ")
            # Assuming each page ends with a form feed character '\f'
            texts = full_text.split('\f')
            # Optionally, strip each page's text to remove unwanted leading/trailing whitespace
            texts = [text.strip() for text in texts if text.strip() != '']

        except Exception as e:
            logging.error(f"Error extracting text from {file_path}: {e}")

        return texts

    @staticmethod
    def preprocess(text):
        """Preprocesses a given text."""
        tokens = [token.lower() for token in word_tokenize(text) if token.isalpha()]
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]

        lemmatizer = WordNetLemmatizer()
        pos_tags = nltk.pos_tag(tokens)
        tokens = [lemmatizer.lemmatize(token, PDFProcessor._get_wordnet_pos(pos_tag)) for token, pos_tag in pos_tags]
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
    def train_doc2vec_model(docs, vector_size=50, window=5, min_count=2, workers=4, epochs=100):
        """Train a Doc2Vec model with the provided documents."""
        tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(docs)]
        model = Doc2Vec(vector_size=vector_size, window=window, min_count=min_count, workers=workers)
        model.build_vocab(tagged_data)
        model.train(tagged_data, total_examples=model.corpus_count, epochs=epochs)
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
        return [PDFProcessor.preprocess(page) for page in pages]

    def build(self, directory_path):
        """Builds the index from a given directory of PDF files page by page."""
        file_paths = glob.glob(os.path.join(directory_path, '*.pdf'))

        with Pool(cpu_count()) as pool:
            processed_pages = [page for doc in tqdm(pool.imap_unordered(self._process_file, file_paths), total=len(file_paths)) for page in doc]

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(processed_pages)

        data = {
            'vectorizer': vectorizer,
            'document_pages': [(fp, idx) for fp in file_paths for idx in range(len(PDFProcessor.extract_text_by_page(fp)))],
        }

        if self.mode == "lsi":
            lsi_model = TruncatedSVD(n_components=50)
            data['lsi_matrix'] = lsi_model.fit_transform(tfidf_matrix)
            data['lsi_model'] = lsi_model
        elif self.mode == "doc2vec":
            d2v_processor = Doc2VecProcessor()
            self.d2v_model = d2v_processor.train_doc2vec_model(processed_pages)
            data['d2v_model'] = self.d2v_model
            data['document_vectors'] = [self.d2v_model.dv[i] for i in range(len(processed_pages))]
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

        if self.mode == "lsi":
            query_vector = self.data['vectorizer'].transform([preprocessed_query])
            lsi_query_vector = self.data['lsi_model'].transform(query_vector)
            similarities = cosine_similarity(self.data['lsi_matrix'], lsi_query_vector).flatten()
        elif self.mode == "doc2vec":
            query_vector = Doc2VecProcessor.infer_vector(self.d2v_model, preprocessed_query)
            scores = cosine_similarity([query_vector], self.data['document_vectors'])
            similarities = scores[0]
        else:
            query_vector = self.data['vectorizer'].transform([preprocessed_query])
            similarities = cosine_similarity(self.data['tfidf_matrix'], query_vector).flatten()
            print(self.data['tfidf_matrix'])
        print(similarities)
        top_indices = similarities.argsort()[:-top_k - 1:-1]
        print(top_indices)
        scores = similarities[top_indices]
        paths = [(self.data['document_pages'][index][0], self.data['document_pages'][index][1]) for index in top_indices]

        return scores, paths


def main():
    """Main function for the command-line interface of the PDF search engine."""
    # Argparse setup
    parser = argparse.ArgumentParser(description='PDF Search Engine')
    parser.add_argument('--index', type=str, default='index_data.pkl', help='Path to the index file.')
    parser.add_argument('--docs', type=str, default='docs', help='Path to the directory containing PDF documents.')
    parser.add_argument('--update-index', action='store_true', help='Update the index if it already exists.')
    parser.add_argument('--mode', type=str, choices=['tfidf', 'lsi', 'doc2vec'], default='tfidf', help='The indexing and search mode.')

    args = parser.parse_args()

    # Create or load index data
    if not args.update_index and os.path.exists(args.index):
        with open(args.index, 'rb') as file:
            index_data = pickle.load(file)
    else:
        index_builder = IndexBuilder(args.mode)
        index_data = index_builder.build(args.docs)

        with open(args.index, 'wb') as file:
            pickle.dump(index_data, file)

    search_engine = SearchEngine(index_data, args.mode)
    while True:
        text = input("Enter your query: ").strip()
        if not text:
            break

        scores, paths = search_engine.query(text)
        for score, (doc_path, page_num) in zip(scores, paths):
            print(f"Document: {doc_path}, Page: {page_num}, Similarity Score: {score:.4f}")


if __name__ == "__main__":
    main()
