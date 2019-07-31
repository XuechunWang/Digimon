import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import gensim.models.keyedvectors as word2vec
import math

from gingerit.gingerit import GingerIt
import nltk
nltk.download('punkt')

def essay_to_wordlist(essay_v, remove_stopwords):
    """Remove the tagged labels and word tokenize the sentence."""
    essay_v = re.sub("[^a-zA-Z]", " ", essay_v)
    words = essay_v.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return (words)

def essay_to_sentences(essay_v, remove_stopwords):
    """Sentence tokenize the essay and call essay_to_wordlist() for word tokenization."""
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences = tokenizer.tokenize(essay_v.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(essay_to_wordlist(raw_sentence, remove_stopwords))
    return sentences

def makeFeatureVec(words, model, num_features):
    """Make Feature Vector from the words list of an Essay."""
    featureVec = np.zeros((num_features,),dtype="float32")
    num_words = 0.
    index2word_set = set(model.wv.index2word)
    for word in words:
        if word in index2word_set:
            num_words += 1
            featureVec = np.add(featureVec,model[word])
    featureVec = np.divide(featureVec,num_words)
    return featureVec

def getAvgFeatureVecs(essays, model, num_features):
    """Main function to generate the word vectors for word2vec model."""
    counter = 0
    essayFeatureVecs = np.zeros((len(essays),num_features),dtype="float32")
    for essay in essays:
        essayFeatureVecs[counter] = makeFeatureVec(essay, model, num_features)
        counter = counter + 1
    return essayFeatureVecs


## Reserved for functions:
def essay_to_sentences(essay_v):
    """Sentence tokenize the essay and call essay_to_wordlist() for word tokenization."""
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences = tokenizer.tokenize(essay_v.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(raw_sentence)
    return sentences

def correct_sentences(sentences):
    parser = GingerIt()
    corrections = []
    for s in sentences:
        feedback = parser.parse(s)
        corrections.append(feedback)
    return corrections

def show_feedback(corrections):
    feedback = ""
    for c in corrections:
        if (c.get('corrections') == []) is False:
            cor = c.get("corrections")[0]
            feedback = feedback + c.get("text") + "\n"
            feedback = feedback + ("-" * 60) + "\n"
            feedback = feedback + ("建议: {}" + "\n" + "错误: {}" + "\n" +
                  "修改意见: {}" + "\n" +
                  "错词的定义: {}").format(c.get("result"), cor.get('text'),
                                           cor.get('correct'), cor.get('definition'))
            feedback = feedback + ("\n")
        else:
            feedback = feedback + (c.get("text")) + "\n"
            feedback = feedback + ("-" * 60) + "\n"
            feedback = feedback + ("没有错误，好样的！\n")
    return feedback
