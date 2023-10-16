from bs4 import BeautifulSoup
from nltk.tokenize.toktok import ToktokTokenizer
import re
import nltk
import emosent as em
from string import punctuation
import spacy
from nltk.corpus import wordnet
import unicodedata


em.EMOJI_SENTIMENT_DICT['❤️'] = {'unicode_codepoint': '0x2764', 'occurrences': 8050, 'position': 0.746943086, 'negative': 355.0, 'neutral': 1334.0, 'positive': 6361.0, 'unicode_name': 'HEAVY BLACK HEART', 'unicode_block': 'Dingbats', 'sentiment_score': 0.746}

emoji_list = list(em.EMOJI_SENTIMENT_DICT.keys())

tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('portuguese') + list(punctuation)
stopword_list.remove("não")
# stopword_list.remove("nunca")
nlp = spacy.load('pt_core_news_sm')
# nlp_vec = spacy.load('en_vectors_web_lg', parse=True, tag=True, entity=True)


def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    if bool(soup.find()):
        [s.extract() for s in soup(['iframe', 'script'])]
        stripped_text = soup.get_text()
        stripped_text = re.sub(r'[\r|\n|\r\n]+', '\n', stripped_text)
    else:
        stripped_text = text
    return stripped_text


def remove_twitter_handles(text):
    r = re.findall("@[\w]*", text)
    for i in r:
        text = re.sub(i, '', text)
    return text


def remove_url_links(text):
    r = re.findall("https?://[A-Za-z0-9./]*", text)
    for i in r:
        text = re.sub(i, '', text)
    return text


def remove_twitter_hashtag(text):
    r = re.findall("#[\w]*", text)
    for i in r:
        text = re.sub(i, '', text)
    return text


def simple_porter_stemming(text):
    ps = nltk.porter.PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])
    return text


def lemmatize_text(text):
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text


def remove_repeated_characters(text):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    repeat_pattern = re.compile(r'(\w*)(\w)\2(\w*)')
    match_substitution = r'\1\2\3'

    def replace(old_word):
        if wordnet.synsets(old_word):
            return old_word
        new_word = repeat_pattern.sub(match_substitution, old_word)
        return replace(new_word) if new_word != old_word else new_word

    correct_tokens = [replace(word) for word in tokens]
    filtered_text = ' '.join(correct_tokens)
    return filtered_text


def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text


def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-Z0-9\s]|\[|\]' if not remove_digits else r'[^a-zA-Z\s]|\[|\]'
    text = re.sub(pattern, '', text)
    return text


def remove_stopwords(text, is_lower_case=False, stopwords=stopword_list):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopwords]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopwords]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text


def normalize_corpus(corpus, html_stripping=True,
                     accented_char_removal=True, text_lower_case=True,
                     text_stemming=False, text_lemmatization=True,
                     special_char_removal=True, remove_digits=True,
                     stopword_removal=True, remove_repeated_char=True,
                     remove_urls=True, remove_tt_handles=True, remove_tt_hashtag=True,
                     stopwords=stopword_list):
    normalized_corpus = []
    # normalize each document in the corpus
    for doc in corpus:
        # strip HTML
        if html_stripping:
            doc = strip_html_tags(doc)

        # removing urls
        if remove_urls:
            doc = remove_url_links(doc)

        # removing twitter handles (@user)
        if remove_tt_handles:
            doc = remove_twitter_handles(doc)

        if remove_tt_hashtag:
            doc = remove_twitter_hashtag(doc)

        # remove extra newlines
        doc = doc.translate(doc.maketrans("\n\t\r", "   "))

        if remove_repeated_char:
            doc = remove_repeated_characters(doc)

        # remove accented characters
        if accented_char_removal:
            doc = remove_accented_chars(doc)

        # lemmatize text
        if text_lemmatization:
            doc = lemmatize_text(doc)

        # stem text
        if text_stemming and not text_lemmatization:
            doc = simple_porter_stemming(doc)

        if special_char_removal:
            # insert spaces between special characters to isolate them
            special_char_pattern = re.compile(r'([{.(-)!}])')
            doc = special_char_pattern.sub(" \\1 ", doc)
            doc = remove_special_characters(doc, remove_digits=remove_digits)
        #             doc = remove_special_characters(doc)

        # remove extra whitespace
        doc = re.sub(' +', ' ', doc)

        # lowercase the text
        if text_lower_case:
            doc = doc.lower()

        # remove stopwords
        if stopword_removal:
            doc = remove_stopwords(doc, is_lower_case=text_lower_case, stopwords=stopwords)

        # remove extra whitespace
        doc = re.sub(' +', ' ', doc)
        doc = doc.strip()

        normalized_corpus.append(doc)

    return normalized_corpus


# -----------------------------------
# ----- DEFINING EMOJI FEATURES -----
# -----------------------------------

def create_emoji_features(df):
    df['emojis'] = ""
    for t in df.index:
        tweet = df.loc[t]
        em_list = ""
        text = tweet['text']
        word_list = text.split(' ')
        for w in word_list:
            if w in emoji_list:
                em_list = em_list + em.get_emoji_sentiment_rank(w)['unicode_codepoint'] + ' '
        df.at[t, 'emojis'] = em_list
    return df