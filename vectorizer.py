from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def tfidf_vectorizer(corpus, ngram):
    print("Vectorizing with TF-IDF...")
    tv = TfidfVectorizer(ngram_range=ngram)
    X = tv.fit_transform(corpus['text_clean'])
    y = corpus['coders_classification']

    return [X, y]