import email
import glob
import math
import random
import re
from collections import Counter, defaultdict

from bs4 import BeautifulSoup
from dynaconf import settings  # Change: Reads features flags from settings.toml file
from nltk import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from machine_learning import split_data


def tokenize(message):
    message = message.lower()  # convert to lowercase

    if settings.NLTK_TOKENIZE:
        # Change: Extract words using function from NLTK module
        all_words = word_tokenize(message)
    else:
        all_words = re.findall("[a-z0-9']+", message)  # extract the words

    if settings.MIN_COUNT:
        # Change: consider only repeated words a minimum number of times
        counter = Counter(all_words)
        all_words = [w for w in all_words if counter[w] >= 2]

    if settings.SPECIAL_TOKENS:
        # Change: add special tokens in words list
        contains_number = any(x.isdigit() for x in message)
        if contains_number:
            all_words.append("contains:number")

    if settings.STEM_WORDS:
        # Change: contract words using porter stemmer from NLTK
        ps = PorterStemmer()
        all_words = {ps.stem(w) for w in all_words}

    if settings.REMOVE_STOP_WORDS:
        # Change: remove english stop words
        stop_words = set(stopwords.words("english"))
        all_words = (w for w in all_words if w not in stop_words)
    return set(all_words)  # remove duplicates


def count_words(training_set):
    """training set consists of pairs (message, is_spam)"""
    counts = defaultdict(lambda: [0, 0])
    for message, is_spam in training_set:
        for word in tokenize(message):
            counts[word][0 if is_spam else 1] += 1
    return counts


def word_probabilities(counts, total_spams, total_non_spams, k=0.5):
    """turn the word_counts into a list of triplets
    w, p(w | spam) and p(w | ~spam)"""
    return [
        (
            w,
            (spam + k) / (total_spams + 2 * k),
            (non_spam + k) / (total_non_spams + 2 * k),
        )
        for w, (spam, non_spam) in counts.items()
    ]


def spam_probability(word_probs, message):
    message_words = tokenize(message)
    log_prob_if_spam = log_prob_if_not_spam = 0.0

    for word, prob_if_spam, prob_if_not_spam in word_probs:

        # for each word in the message,
        # add the log probability of seeing it
        if word in message_words:
            log_prob_if_spam += math.log(prob_if_spam)
            log_prob_if_not_spam += math.log(prob_if_not_spam)

        # for each word that's not in the message
        # add the log probability of _not_ seeing it
        else:
            log_prob_if_spam += math.log(1.0 - prob_if_spam)
            log_prob_if_not_spam += math.log(1.0 - prob_if_not_spam)

    prob_if_spam = math.exp(log_prob_if_spam)
    prob_if_not_spam = math.exp(log_prob_if_not_spam)
    try:
        return prob_if_spam / (prob_if_spam + prob_if_not_spam)
    except ZeroDivisionError:
        # Change: adds treatement for ZeroDivisionError exception
        return prob_if_spam > (prob_if_spam + prob_if_not_spam)


class NaiveBayesClassifier:
    def __init__(self, k=0.5):
        self.k = k
        self.word_probs = []

    def train(self, training_set):

        # count spam and non-spam messages
        num_spams = len([is_spam for message, is_spam in training_set if is_spam])
        num_non_spams = len(training_set) - num_spams

        # run training data through our "pipeline"
        word_counts = count_words(training_set)
        self.word_probs = word_probabilities(
            word_counts, num_spams, num_non_spams, self.k
        )

    def classify(self, message):
        return spam_probability(self.word_probs, message)


def get_email_body(row_email: str) -> str:
    # Change: Implements function to extract messages from email body payload
    email.message_from_string(row_email)
    b = email.message_from_string(row_email)
    payloads = []
    if b.is_multipart():
        for part in b.get_payload():
            payloads.append(part)
    # not multipart - i.e. plain text, no attachments, keeping fingers crossed
    else:
        payloads.append(b)

    body = ""
    for p in payloads:
        ctype = p.get_content_type()
        if ctype == "text/plain":
            body += p.get_payload()
        elif ctype == "text/html":
            soup = BeautifulSoup(p.get_payload(), "html.parser")
            # kill all script and style elements
            for script in soup(["script", "style"]):
                script.extract()  # rip it out

            # get text
            text = soup.get_text()

            # break into lines and remove leading and trailing space on each
            lines = (line.strip() for line in text.splitlines())
            # break multi-headlines into a line each
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            # drop blank lines
            body = " ".join(chunk for chunk in chunks if chunk)
    return body


def get_subject_data(path):

    data = []

    # regex for stripping out the leading "Subject:" and any spaces after it
    subject_regex = re.compile(r"^Subject:\s+")

    # Change: Implements another regex patterns to filter new terms
    from_regex = re.compile(r"^From:\s+")
    received_regex = re.compile(r"^Received:\s+")
    url_regex = re.compile(r"^Url:\s+")

    # glob.glob returns every filename that matches the wildcarded path
    for fn in glob.glob(path):
        is_spam = "ham" not in fn

        message = ""  # Change: Gets only one message per email

        if settings.CONTENT_FROM_BODY:
            # Change: get message from email body
            with open(fn, "r", encoding="ISO-8859-1") as file:
                message += " " + get_email_body(file.read())

        with open(fn, "r", encoding="ISO-8859-1") as file:
            for line in file:
                if line.startswith("Subject:"):
                    message += " " + subject_regex.sub("", line).strip()
                if settings.ADD_FROM:  # Change: Get content from "From:" fields
                    if line.startswith("From:"):
                        message += " " + from_regex.sub("", line).strip()
                if settings.ADD_URL:  # Change: Get content from "Url:" fields
                    if line.startswith("Url:"):
                        message += " " + url_regex.sub("", line).strip()
                if settings.ADD_RECEIVED:  # Change: Get content from "Reveived:" fields
                    if line.startswith("Received:"):
                        message += " " + received_regex.sub("", line).strip()
            if message:
                data.append((message, is_spam))

    return data


def p_spam_given_word(word_prob):
    word, prob_if_spam, prob_if_not_spam = word_prob
    return prob_if_spam / (prob_if_spam + prob_if_not_spam)


def train_and_test_model(path, settings_options: dict = None):
    if settings_options:
        for key, value in settings_options.items():
            setattr(settings, key, value)
    data = get_subject_data(path)
    random.seed(0)  # just so you get the same answers as me
    train_data, test_data = split_data(data, 0.83)  # Change: Use 0.83 percentage to split between train and test data

    classifier = NaiveBayesClassifier()
    classifier.train(train_data)

    classified = [
        (subject, is_spam, classifier.classify(subject))
        for subject, is_spam in test_data
    ]

    counts = Counter(
        (is_spam, spam_prob > 0.5)  # (actual, predicted)
        for _, is_spam, spam_prob in classified
    )

    # Change: Adds accuracy as metric to compare results
    hits, misses = counts[(True, True)] + counts[(False, False)], counts[(True, False)] + counts[(False, True)]
    accuracy = hits/len(classified)
    print("Accuracy:", accuracy, "Counts:", counts, "Settings:", settings_options)

    # classified.sort(key=lambda row: row[2])
    # spammiest_hams = list(filter(lambda row: not row[1], classified))[-5:]
    # hammiest_spams = list(filter(lambda row: row[1], classified))[:5]
    #
    # print("spammiest_hams", spammiest_hams)
    # print("hammiest_spams", hammiest_spams)
    #
    # words = sorted(classifier.word_probs, key=p_spam_given_word)
    #
    # spammiest_words = words[-5:]
    # hammiest_words = words[:5]
    #
    # print("spammiest_words", spammiest_words)
    # print("hammiest_words", hammiest_words)
    return accuracy


if __name__ == "__main__":
    train_and_test_model(r"./emails/*/*")
