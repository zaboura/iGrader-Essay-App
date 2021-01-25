# import libraries
from __future__ import print_function
from __future__ import division
#from nltk.stem import WordNetLemmatizer
from nltk import ngrams, word_tokenize
#from gensim.models import CoherenceModel
from textblob import TextBlob
#from spacy.lang.en.stop_words import STOP_WORDS
from operator import truediv
from transformers import BertTokenizer, BertModel
from scipy.spatial import distance
from pytorch_transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch.nn.functional as F
from readability import Readability
from spacy.matcher import PhraseMatcher
from numpy import dot
from numpy.linalg import norm
#import language_check
import faulthandler; faulthandler.enable()
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import spacy
import nltk
import warnings
import gensim
import torch
import tensorflow as tf
import tensorflow_hub as hub
import string
import gensim.corpora as corpora
import neuralcoref
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess

#nltk.download('stopwords')
nlp = spacy.load('en_core_web_lg')
stop_words = stopwords.words('english')
neuralcoref.add_to_pipe(nlp)

# Setup Pandas
# pd.set_option('display.width', 500)
# pd.set_option('display.max_columns', 100)
# pd.set_option('display.notebook_repr_html', True)
# pd.set_option('display.max_colwidth', 100)

warnings.simplefilter("ignore", DeprecationWarning)

encoder_model = hub.load("models/universal-sentence-encoder-large_5")

# Load pre-trained model BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load pre-trained model tokenizer (vocabulary)
tokenizerGPT = GPT2Tokenizer.from_pretrained('gpt2')

# Load pre-trained model (weights)
modelGPT = GPT2LMHeadModel.from_pretrained('gpt2')

# Set the model in evaluation mode to deactivate the DropOut modules
modelGPT.eval()


# Load pre-trained model (weights)
model_BERT = BertModel.from_pretrained('bert-base-uncased',
                                      # Whether the model returns all hidden-states.
                                      output_hidden_states=True,
                                      )
# Put the model in "evaluation" mode, meaning feed-forward operation.
model_BERT.eval()

transition_words = ["in the first place", "again", "moreover", "to", "as well as", "as a matter of fact", "and", "together with",
                    "in like manner", "also", "of course", "in addition", "then", "likewise","coupled with", "equally",
                    "comparatively","in the same fashion", "in the same way", "identically", "correspondingly","first",
                    "second", "third", "uniquely", "similarly","in the light of", "like", "furthermore","not to mention",
                    "as", "additionally","to say nothing of", "too","equally important","by the same token","in other words",
                    "notably", "in fact","to put it differently", "including", "in general","for one thing", "like", "in particular",
                    "as an illustration", "to be sure", "in detail","in this case", "namely", "to demonstrate","for this reason",
                    "chiefly", "to emphasize","to put it another way", "truly", "to repeat","that is to say", "indeed", "to clarify",
                    "with attention to", "certainly", "to explain","by all means", "surely", "to enumerate","important to realize",
                    "markedly", "such as","another key point", "especially for", "example","first thing to remember", "specifically",
                    "for instance","most compelling evidence", "expressively", "to point out","must be remembered", "surprisingly",
                    "with this in mind","point often overlooked", "frequently","on the negative side", "significantly",
                    "on the positive side","in the middle", "here", "further","to the left", "to the right", "there", "beyond",
                    "in front of", "next", "nearby","on this side", "where", "wherever","in the distance", "from", "around",
                    "here and there", "over", "before","in the foreground", "near", "alongside","in the background", "above",
                    "amid","in the center of", "below", "among","down", "beneath","adjacent to", "up", "beside","opposite to",
                    "under", "behind","between", "across","at the present time", "after", "henceforth","from time to time", "later",
                    "whenever","sooner or later", "last", "eventually","at the same time", "until", "meanwhile","up to the present time",
                    "till", "further","to begin with", "since", "during","in due time", "then", "first", "second","until now", "before",
                    "in time","as soon as", "hence", "prior to", "forthwith","in the meantime", "when", "straightaway",
                    "in a moment", "once","without delay", "about", "by the time","in the first place", "next", "whenever","all of a sudden",
                    "now","at this instant", "now that","immediately", "formerly", "instantly","quickly", "suddenly", "presently",
                    "finally", "shortly", "occasionally","as a result", "for", "consequently","under those circumstances", "thus",
                    "therefore","in that case", "because the", "thereupon","for this reason", "then", "forthwith","henceforth",
                    "hence", "accordingly","although this may be true", "but", "although","in contrast", "still", "and still",
                    "instead","different from", "unlike", "whereas","of course, but", "or", "despite","on the other hand", "and yet",
                    "conversely","on the contrary", "while", "otherwise","at the same time", "albeit", "however","in spite of", "besides",
                    "rather","even so", "nevertheless","be that as it may", "even though", "nonetheless",
                    "then again", "regardless","above all", "notwithstanding","in reality","after all","in the event that", "if",
                    "in case","granted that", "then", "provided that","as long as", "unless", "given that","on condition that",
                    "only if", "even if","for the purpose of", "when", "so that","with this intention", "whenever", "so as to",
                    "with this in mind", "since", "owing to","in the hope that", "while", "due to","to the end that","for fear that",
                    "because of", "as much as","in order to", "as","seeing that", "being that", "since","in view of", "while","lest",
                    "as can be seen", "after all", "overall","generally speaking", "in fact", "ordinarily","in the final analysis",
                    "in summary", "usually","all things considered", "in conclusion", "by and large","as shown above", "in short",
                    "to sum up","in the long run", "in brief", "on the whole","given these points", "in essence", "in any event",
                    "as has been noted", "to summarize", "in either case","in a word", "on balance", "all in all","for the most part",
                    "altogether","than", "that", "after","rather than", "what", "as long as","whether", "whatever", "as soon as",
                    "as much as", "which", "before","whereas", "whichever", "by the time","now that","though",  "who", "once","although",
                    "whoever", "since","even though", "whom","while", "whomever", "until","whose", "when","if", "where", "whenever",
                    "only if", "wherever","unless","until", "how", "because","provided that", "as though", "since","assuming that",
                    "as if", "so that","even if", "in order","in case that", "why", "what with and","just as so", "whether or","both and"]



def words_count(input_data, input_feature_name):
    
    """count number of words

    Parameters
    ----------
    input_df : dataframe
        the input dataframe containing 'essay' feature
    input_feature_name : str
        feature name wanted to count number of it words

    Returns
    -------
    list
        a list of numbers of words for each essay
    
    """
    
    #empty list
    nbr_tokens = []

    for essay in input_data[input_feature_name]:
        nbr_tokens.append(len(essay.split())) 
        
    return nbr_tokens


def sentences(input_data, feature_name):
    
    """Extract the sentences of the essay

    Parameters
    ----------
    input_df : dataframe
        the input dataframe containing 'essay' feature
    input_feature_name : str
        feature name wanted to extract sentences from

    Returns
    -------
    list
        a list of sentences of each essay
    
    """
    
    #empty list
    sentences = []

    for essay in input_data[feature_name]:
        sentences.append([sen for sen in re.split('\.|!|\?', essay) if len(sen) > 2])
    
    return sentences


def sents_count(input_data, input_feature_name):
    
    """count number of sentences of the essay

    Parameters
    ----------
    input_df : dataframe
        the input dataframe containing 'essay' feature
    input_feature_name : str
        feature name wanted to count sentences for

    Returns
    -------
    list
        a list of numbers of sentences of each essay
    
    """
    
    #empty list
    nbr_sents = []

    for essay in input_data[input_feature_name]:
        nbr_sents.append(len([sent for sent in nltk.sent_tokenize(essay)]))
    
    return nbr_sents 
    

def avrg_sents_length(input_data, input_feature_name):
    
    
    """extract average length of sentences of the essay

    Parameters
    ----------
    input_df : dataframe
        the input dataframe containing 'essay' feature
    input_feature_name : str
        feature name wanted to be processed

    Returns
    -------
    list
        a list average length of sentences of each essay
    
    """
    
    #empty lists
    nbr_sents = []
    nbr_tokens = []
    
    for essay in nlp.pipe(input_data[input_feature_name], batch_size=100):
        nbr_sents.append(len([sent.string.strip() for sent in essay.sents]))
        nbr_tokens.append(len([e.text for e in essay]))
    
    return [int(item) for item in list(map(truediv,nbr_tokens , nbr_sents))] # create new feature in data frame
    

def topic_detection_(input_df):
    
    """detect the topic of the essay

    Parameters
    ----------
    input_df : dataframe
        the input dataframe containing 'essay' feature

    Returns
    -------
    list
        a list of topics for essays
    
    """
    
    topic_detection = []
    
    for essay in input_df['corrected']:
        topic_detection.append(topic_detection_lda([essay]))
        
    return topic_detection


def lexical_divercity_(input_df):
    
    
    """detect lexical diversity of the essay

    Parameters
    ----------
    input_df : dataframe
        the input dataframe containing 'essay' feature

    Returns
    -------
    list
        a list of lexical divercity 
    
    """
    
    lexical_divr = []
    
    for essay in input_df['corrected']:
        lexical_divr.append(lexical_diversity(essay))
        
    return lexical_divr        
        
def FK_score(input_df):
    
    """compute fk-score of the essay

    Parameters
    ----------
    input_df : dataframe
        the input dataframe containing 'essay' feature

    Returns
    -------
    list
        a list of scores
    
    """
    
    #empty list
    fk_score = []
    for essay in input_df['corrected']:
        #set a threshold for number of tokens in the essay
        if len(re.findall(r'\w+', essay)) >= 110:
            fk_score.append(flesch_kincaid_score(essay))
        else:
            fk_score.append(0)
            
    return fk_score


# helper functions
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))


def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]


def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append(
            [token.lemma_ for token in doc if token.pos_ in allowed_postags])

    return texts_out



def topic_detection_lda(essay):
    

    id2word, texts, corpus = preprocessing_for_lda(essay)
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=1,
                                                random_state=100,
                                                update_every=1,
                                                chunksize=100,
                                                passes=10,
                                                alpha='auto',

                                                per_word_topics=True)
    topic = lda_model.show_topic(0)
    topic_words = extract_main_words_from_topic(topic)
    
    return topic_words



def preprocessing_for_lda(essay):
    """Returns the dictionary, corpus and its term-document frequency representation."""
    data_words = list(sent_to_words(essay))
    data_words_nostops = remove_stopwords(data_words)
    data_lemmatized = lemmatization(data_words_nostops, allowed_postags=[
                                    'NOUN', 'ADJ', 'VERB', 'ADV'])
    id2word = corpora.Dictionary(data_lemmatized)  # create dictionary
    texts = data_words  # create corpus
    corpus = [id2word.doc2bow(text)
              for text in texts]  # term-document frequency
    return id2word, texts, corpus


def extract_main_words_from_topic(topic):
    
    words_list = []
    for i in range(len(topic)):
        words_list.append(topic[i][0])
        
    return words_list


def lexical_diversity(essay):
    return len(set(essay)) / len(essay)


# get the vocab size of the essays
def get_vocabulary_size(essay):
    
    vocab = set(w.lower() for w in essay if w.isalpha())
    
    return len(vocab)

def flesch_kincaid_score(essay):
    
    r = Readability(essay)
    f = r.flesch_kincaid()
    
    return f.score


def BERT_Embedding(data, feature_name):

    embeddings = []
    for sentences in data[feature_name]:
        
        essay_sentences = []
        
        for text in sentences:
            
            # Add the special tokens.
            marked_text = "[CLS] " + text + " [SEP]"

            # Split the sentence into tokens.
            tokenized_text = tokenizer.tokenize(marked_text)

            # Map the token strings to their vocabulary indeces.
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

            # Mark each of the tokens as belonging to sentence "1".
            segments_ids = [1] * len(tokenized_text)

            # Convert inputs to PyTorch tensors
            tokens_tensor = torch.tensor([indexed_tokens])
            segments_tensors = torch.tensor([segments_ids])

            # Run the text through BERT, and collect all of the hidden states produced
            # from all 12 layers.
            with torch.no_grad():

                outputs = model_BERT(tokens_tensor, segments_tensors)

                # Evaluating the model will return a different number of objects based on
                # how it's  configured in the `from_pretrained` call earlier. In this case,
                # becase we set `output_hidden_states = True`, the third item will be the
                # hidden states from all layers.
                hidden_states = outputs[2]

            # `token_vecs` is a tensor with shape [#words x 768]
            token_vecs = hidden_states[-2][0]

            # Calculate the average of all token vectors.
            sentence_embedding = torch.mean(token_vecs, dim=0)
            essay_sentences.append(sentence_embedding)

        embeddings.append(essay_sentences)
        

    return embeddings



def essay_similarity(df, feature_name):
    """
    computes the mean of cosine similarity between all possible combination of sentences embedding
    
    Parameters: 
    -----------
         df: dataframe
            input data frame 
         feaature_name: str
            essay's sentences 
    Returns:
    --------
        list:
            mean of cosine similarities of all sentences
    """
    
    similarity = []

    embeddings = BERT_Embedding(df, feature_name)

    for emb in embeddings:
        mean_sim = []
        for i in range(len(emb)-1):
            for j in range(i+1, len(emb)):
                cos_sim = distance.cosine(emb[i][0],emb[j][0])
                mean_sim.append(cos_sim)
        similarity.append(np.mean(mean_sim))        
            
    return similarity
    

def tree_height(root):
    
    """Returns the maximum depth or height of the dependency parse of a sentence."""
    
    
    if not list(root.children):
        return 1
    else:
        return 1 + max(tree_height(x) for x in root.children)


def average_tree_height(text):
    
    
    """Computes average height of parse trees for each sentence in a text."""
    
    
    if type(text) == str:
        doc = nlp(text)
    else:
        doc = text
    roots = [sent.root for sent in doc.sents]
    return np.mean([tree_height(root) for root in roots])


def avg_tree_height(df, feature_name):
    """
    
    Returns average tree height as appended column to the df for 
    all the rows of the input dataframe where the essays are stored in column name 'essays'.
    
    """
    Avg_tree_height = df[feature_name].apply(average_tree_height)
    return Avg_tree_height


def polarity_with_tb(text):
    """Returns polarity score as detected with TextBlob. """
    polarity = TextBlob(text).sentiment.polarity
    return polarity


def polarity(df, feature_name):
    
    """Returns polarity as appended column to the df for all the rows of the input dataframe where 
    the essays are stored in column name 'essays'."""
    
    polarity = df[feature_name].apply(polarity_with_tb)
    return polarity

def subjectivity_with_tb(text):
    
    """Returns polarity score as detected with TextBlob."""
    
    subjectivity = TextBlob(text).sentiment.subjectivity
    return subjectivity


def subjectivity(df, feature_name):
    
    """Returns polarity as appended column to the df for all the rows of the input dataframe
    where the essays are stored in column name 'essays'."""
    
    subjectivity = df[feature_name].apply(subjectivity_with_tb)
    return subjectivity




def sentence_coherence(text, mean_prob_value):
    #compute the encoding of each token
    indexed_tokens = tokenizerGPT.encode(text)
    
    word_probabilities = []
    contexts = [indexed_tokens[:i+1] for i in range(len(indexed_tokens))][1:]
    

    for context in contexts:
        
        indexed_tokens = context[:-1]
    
        # Convert indexed tokens in a PyTorch tensor
        tokens_tensor = torch.tensor([indexed_tokens])

        # If you have a GPU, put everything on cuda
        # tokens_tensor = tokens_tensor.to('cuda')
        # model.to('cuda')

        # Predict all tokens
        with torch.no_grad():
            outputs = modelGPT(tokens_tensor)
            predictions = outputs[0]
        
        # the output vector, each case correspond to a kind of probabilitie, 
        # and the corresponding index of the case to a word
        ss = torch.sort(predictions[0, -1, :]).values

        # The softmax make the sum of tensor values to one
        probs = F.softmax(predictions[0, -1, :], dim=0)
        
        ## predicted_text = tokenizer.decode(indexed_tokens + [context[-1]])
        
        #probability of the word
        prob_value = float(probs[context[-1]])
        #customized_prob_value = -np.log(prob_value)/np.log(prob_value)**2
        
        word_probabilities.append(prob_value)
    
    #print(np.prod(word_probabilities))

    return np.prod(word_probabilities)/mean_prob_value**(len(indexed_tokens)-2)


def text_coherence(text, mean_word_prob):
    # split into sentences
    sentences = text.split('.')

    # remove empty sentences and too small sentences
    sentences = [sent for sent in sentences if len(sent) > 3]

    return np.mean([sentence_coherence(sentence, mean_word_prob) for sentence in sentences])


def text_coherence_DF(DF, text_column_name):
    
    all_sentences = ' '.join([row[text_column_name] for index, row in DF.iterrows()])
    mean_word_prob = 1/5000
    DF['text_coherence'] = DF[text_column_name].apply(lambda x: text_coherence(x, mean_word_prob))
    return DF




def transition_words_counter(essay):
    matcher = PhraseMatcher(nlp.vocab)
    patterns = [nlp.make_doc(text) for text in transition_words]
    matcher.add("phrase_matcher", None, *patterns)
    text_doc = nlp(essay)
    character_matches = matcher(text_doc)
    m = []
    for match_id, start, end in character_matches:
        span = text_doc[start:end]
        m.append(span.text)
    return len(m)

def apply_transition_words_counter(input_df, input_feature_name):
    res = input_df[input_feature_name].apply(transition_words_counter)

    return res




def rare_words_counter(text):
    text_vocab = set(w.lower() for w in text if w.isalpha())
    english_vocab = set(w.lower() for w in nltk.corpus.words.words())
    unusual = text_vocab - english_vocab
    return len(unusual)


def apply_rare_words_counter(input_df, input_feature_name):
    res = input_df[input_feature_name].apply(rare_words_counter)

    return res

def content_ratio(text):
    content = [w for w in text if w.lower() not in stop_words]     
    return len(content) / len(text)

def apply_content_ratio(input_df, input_feature_name):
    res = input_df[input_feature_name].apply(content_ratio)

    return res


def coreference_detector(essay):
    doc = nlp(essay)
    if len(doc._.coref_clusters) != 0:
        res = sum([len(elem) for elem in doc._.coref_clusters])/len(doc._.coref_clusters)
    else:
        res = 0
    return res

def apply_coreference_detector(input_df, input_feature_name):
    res = input_df[input_feature_name].apply(coreference_detector)
    return  res

def distinct_words(essay):
    res = len(set(essay))
    return res

def apply_distinct_words(input_df, input_feature_name):
    res = input_df[input_feature_name].apply(distinct_words)

    return res

def inverse_class_labels_reassign(score):
    
    if score == [1,0,0,0,0]:
        return 1
    elif score == [1,1,0,0,0]:
        return 2
    elif score == [1,1,1,0,0]:
        return 3
    elif score == [1,1,1,1,0]:
        return 4
    else:
        return 5




def relevance_to_prompt(essay, prompt, model):
    embedded_essay = model([essay]).numpy()
    embedded_prompt = model([prompt]).numpy()
    cos_sim = dot(embedded_essay[0], embedded_prompt[0])/(norm(embedded_essay[0])*norm(embedded_prompt[0]))

    return cos_sim


def apply_relevance_to_prompt(input_df, essay_column, prompt_column, model):
    res = input_df.apply(lambda x: relevance_to_prompt(x[essay_column], x[prompt_column], model), axis = 1)
    
    return res


def noun_counter(essay):
    tokens = word_tokenize(essay)
    nouns = [word for (word, pos) in nltk.pos_tag(tokens) if(pos[:2] == 'NN')]
    return len(nouns)


def verb_counter(essay):
    tokens = word_tokenize(essay)
    verbs = [word for (word, pos) in nltk.pos_tag(tokens) if(pos[:2] == 'VB')]
    return len(verbs)


def adverb_counter(essay):
    tokens = word_tokenize(essay)
    adverbs = [word for (word, pos) in nltk.pos_tag(tokens) if(pos[:2] == 'RB')]
    return len(adverbs)


def adjective_counter(essay):
    tokens = word_tokenize(essay)
    adjectives = [word for (word, pos) in nltk.pos_tag(tokens) if(pos[:2] == 'JJ')]
    return len(adjectives)


def apply_noun_counter(input_df, input_feature_name):
    res = input_df[input_feature_name].apply(noun_counter)

    return res

def apply_verb_counter(input_df, input_feature_name):
    res = input_df[input_feature_name].apply(verb_counter)

    return res

def apply_adverb_counter(input_df, input_feature_name):
    res = input_df[input_feature_name].apply(adverb_counter)

    return res

def apply_adjective_counter(input_df, input_feature_name):
    res = input_df[input_feature_name].apply(adjective_counter)

    return res