#import libraries
import sys
from pickle import load
from pandas.core.frame import DataFrame
from src import *
import language_check

    
    
def pipeline_(text, prompt):
    
    """Extract the features from the essay

    Parameters
    ----------
    text : str
        The essay content
    prompt : str
        The prompt content

    Returns
    -------
    dataframe
        a dataframe of all extracted features
    int
        number of words in the essay 
    float
        percentage of similarity between the essay and its prompt
    int
        number of mistakes in the essay
    
    """
    
    # create empty dataframe 
    df = pd.DataFrame()
    # same essay and prompt 
    df['essay'] = [text]
    df['Prompt'] = [prompt]

    tool = language_check.LanguageTool('en-US')
    # detect the mistakes and count them
    df['matches'] = df['essay'].apply(lambda txt: tool.check(txt))
    df['corrections'] = nbr_mistakes = df.apply(lambda l: len(l['matches']), axis=1)
    #correct the essay
    df['corrected'] = df.apply(lambda l: language_check.correct(l['essay'], l['matches']), axis=1)
    # count the number of words
    df['word_count'] = word_count = words_count(df, 'essay')
    # extract average sentence length, sentence's count, and sentences
    df['avrg_sents_length'] = avrg_sents_length(df, 'essay')
    df['sents_count'] = sents_count(df, 'essay')
    df['sents'] = sentences(df, 'essay')
    # detect the topic of the essay
    df['topic_detection'] = topic_detection_(df)
    #measure the lexical diversity
    df['lexical_divr'] = lexical_divercity_(df)
    #compute the fk-dcore
    df['fk_score'] = FK_score(df)
    # counting the punctuation
    count = lambda l1, l2: sum([1 for x in l1 if x in l2])
    df['count_punct'] = df.essay.apply(lambda s: count(s, string.punctuation))
    #extract the polarity 
    df['Polarity'] = polarity(df, 'essay')
    #extract the subjectivity
    df['Subjectivity'] = subjectivity(df, 'essay')
     #extract the paverage height of sparce tree
    df['Avg_tree_height'] = avg_tree_height(df, 'essay')
    #coumpute the similarities between sentences
    df['inner_similarities'] = essay_similarity(df, 'sents')
    #measur the coherence of the essay
    df = text_coherence_DF(df, text_column_name='essay')
    #count the number of linking words
    df["Transition_words_count"] = apply_transition_words_counter(df, 'corrected')
    #count the number of rare words
    df["Rare_words_count"] = apply_rare_words_counter(df, 'essay')
    #compute the content ratio
    df["Content_ratio"] = apply_content_ratio(df, 'essay')
    #compute coreference detector
    df["Coreference_detector"] = apply_coreference_detector(df, 'essay')
    #count distinct words
    df["distinct_words"] = apply_distinct_words(df, 'essay')
    #compute the relevance between essay and prompt
    df['Relevance_to_prompt'] = prompt_sim = apply_relevance_to_prompt(df, 'essay', "Prompt", model= encoder_model)
    #count the number of nouns
    df["Nr_of_nouns"] = apply_noun_counter(df, 'essay')
    #count the number of verbs
    df["Nr_of_verbs"] = apply_verb_counter(df, 'essay')
    #count the number of adverbs
    df["Nr_of_adverbs"] = apply_adverb_counter(df, 'essay')
    #count the number of adjectives
    df["Nr_of_adjectives"] = apply_adjective_counter(df, 'essay')

    #define the features
    COLUMNS = ['word_count', 'corrections', 'avrg_sents_length', 'sents_count', 'lexical_divr', 'fk_score','count_punct',
               'Polarity', 'Subjectivity', 'Avg_tree_height', 'inner_similarities', 'text_coherence',
               'Transition_words_count', 'Rare_words_count', 'Content_ratio', 'Coreference_detector', 'distinct_words',
               'Relevance_to_prompt', 'Nr_of_nouns', 'Nr_of_verbs', 'Nr_of_adverbs', 'Nr_of_adjectives']

    #convert the output to numpy
    preprocessed_input = df[COLUMNS].to_numpy().reshape(-1,len(COLUMNS))
    

    return np.array(preprocessed_input), word_count[0], prompt_sim[0], nbr_mistakes[0]

