import os
import time
import re
from multiprocessing import Process, Queue

import numpy as np

import nltk
from nltk.tokenize.moses import MosesDetokenizer

from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options as FirefoxOptions

os.environ['KERAS_BACKEND'] = 'tensorflow'
from textgenrnn import textgenrnn


SLEEP_MINS = 60
TWITTER_USERNAME = 'hypotheticalba1'
TWITTER_PASSWORD = os.environ['HYPBAT_PASSWORD']
CORPUS_FN = 'whowouldwin_questions.cleaned.txt'
WEIGHTS_FN = 'whowouldwin190618.h5'
TEMP_RANGE = (0.75, 1)
MAX_LEN = 125
PREFIX = "Who"
N_CHOICES = 4


# boring or iffy words that we don't want to tweet!
STOPWORDS = set([
    'goku', 'thor', 'superman', 'hulk',
    'spiderman', 'batman', 'black',
    'wonder', 'saitama', 'isis',
    'thanos', 'kkk'
])


# words which are usually part of the sentence context not the combatant names
IGNORE_NNP_WORDS = set([
    'vs', 'vs.', 'combat', 'fight', 'battle',
    'battles', 'war', 'victory', 'edge',
    'rap', 'match', 'winner', 'contest',
    'race', 'duel', 'punch-up', 'top',
    'free', 'brawl', 'victor',
    'victorious', 'champion', 'death'
])


def load_corpus():
    '''
    load the corpus which we use to identify made up words which we like :)
    '''
    corpus = set()
    with open('whowouldwin_questions.cleaned.txt') as f:
        for line in f:
            corpus.update(set(nltk.word_tokenize(line.lower())))
    return corpus


def load_model():
    model = textgenrnn()
    model.load(WEIGHTS_FN)
    return model


detokenize = MosesDetokenizer().detokenize


def parse_fighters(question):
    '''
    find out who's fighting in a question
    '''
    # between is a common word between the question and the fighters
    if 'between' in question.lower():
        question = question.split('between')[1]
    parsed = nltk.word_tokenize(question)
    tagged = nltk.pos_tag(parsed)
    s = None
    e = 0
    # use pos tagging to find nouns and names
    for i, (word, tag) in enumerate(tagged):
        if tag in ['NNP', 'NNPS', 'NN', 'NNS', 'JJ', 'CD'] \
                and word not in IGNORE_NNP_WORDS:
            if s is None:
                s = i
            e = max(e, i + 1)
    if s is None:
        return []
    prev_word = parsed[s - 1].lower()
    if prev_word == 'the' or prev_word == 'a':
        s -= 1
    f = detokenize(
        [w for w, t in zip(parsed[s:e], tagged[s:e])
         if t[1] not in ',.'],
        return_str=True
    )
    # split fighters on "or" or "vs"
    fighters = re.split(
        '(?:\sor\s)|(?:\svs\.?\s)|(?:\sversus\s)', f, re.IGNORECASE)
    # only split on "and" if there are no instances of "or" or "vs"
    if len(fighters) == 1:
        fighters = re.split('(?:\sand\s)', f)
    fighters = [x.replace('.', '') for x in fighters]
    return fighters


def generate_poll(model, corpus, temp=None, max_len=100, prefix=None):
    '''
    use the model to write a question then figure out who's fighting in it
    '''
    while True:
        if temp is None:
            temp = np.random.uniform(0.5, 1)
        try:
            if len(temp) == 2:
                temp = np.random.uniform(temp[0], temp[0])
        except TypeError:
            pass
        question, = model.generate(
            temperature=temp,
            max_gen_length=max_len,
            return_as_list=True,
            prefix=prefix)
        words = set(nltk.word_tokenize(question.lower()))
        if words.issubset(corpus):
            # we want made up words please
            continue
        if words.intersection(STOPWORDS):
            # we don't want boring stopwords like Goku or Thanos thankyou
            continue
        fighters = parse_fighters(question)
        if question.endswith('?') and len(fighters) >= 2:
            return question, fighters


def twitter_login():
    opt = FirefoxOptions()
    opt.add_argument("--headless")
    driver = webdriver.Firefox(firefox_options=opt)
    driver.set_page_load_timeout(30)
    driver.implicitly_wait(30)
    driver.get("https://www.twitter.com/login")

    user_box = driver.find_element_by_class_name("js-username-field")
    user_box.send_keys(TWITTER_USERNAME)

    pass_box = driver.find_element_by_class_name("js-password-field")
    pass_box.send_keys(TWITTER_PASSWORD)
    pass_box.submit()
    return driver


def post_to_twitter(driver, question, options):
    tweetbox = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "tweet-box-home-timeline"))
    )
    tweetbox.send_keys(question)
    poll_btn = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.CLASS_NAME, "PollCreator-btn"))
    )
    poll_btn.click()
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CLASS_NAME, "PollingCardComposer"))
    )
    if len(options) > 2:
        for i in range(len(options) - 2):
            driver.find_element_by_class_name(
                'PollingCardComposer-addOption').click()
    for i, opt in enumerate(options, 1):
        elem = driver.find_element_by_class_name(
            'PollingCardComposer-option{}'.format(i))
        elem = elem.find_element_by_class_name(
             'PollingCardComposer-optionInput')
        if len(opt) > 25:
            opt = opt[:22] + '...'
        elem.send_keys(opt)
    tweet = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.CLASS_NAME, "tweet-action"))
    )
    tweet.click()
    return


def vote_for_best(queue, corpus):
    model = load_model()
    while True:
        print('Started Voting!')
        suggestions = []
        for i in range(1, N_CHOICES + 1):
            question, options = generate_poll(
                model,
                corpus,
                temp=TEMP_RANGE,
                max_len=MAX_LEN,
                prefix=PREFIX
            )
            print('{}.'.format(i))
            print(question)
            for opt in options:
                print('  # {}'.format(opt))
            suggestions.append([question, options])
        while True:
            choice = input('Choose a question\n')
            try:
                choice = int(choice)
            except ValueError:
                print('Not a valid choice, sorry')
                continue
            if 1 <= choice <= N_CHOICES + 1:
                break
            else:
                print('Number is not in range 1-{}'.format(N_CHOICES))
        queue.put(suggestions[choice - 1])


def write_polls(queue, corpus):
    model = load_model()
    # wait for first vote then start posting
    while queue.empty():
        time.sleep(1)
    while True:
        if queue.empty():
            question, opts = generate_poll(
                model,
                corpus,
                temp=TEMP_RANGE,
                max_len=MAX_LEN,
                prefix=PREFIX
            )
        else:
            question, opts = queue.get()
        driver = twitter_login()
        post_to_twitter(driver, question, opts)
        driver.refresh()
        driver.quit()
        time.sleep(60 * SLEEP_MINS)


if __name__ == '__main__':
    corpus = load_corpus()
    queue = Queue(maxsize=1000)
    writing = Process(target=write_polls,
                      args=(queue, corpus))
    writing.daemon = True
    writing.start()
    vote_for_best(queue, corpus)
    writing.join()
