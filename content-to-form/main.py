import nltk
import spacy
import spacy_wordnet
import collections
#import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')
from spacy_wordnet.wordnet_annotator import WordnetAnnotator
from collections import Counter

from nltk.corpus import wordnet as wn

babelnet_ids = {}
nlp = spacy.load("en_core_web_md")
nlp.add_pipe(WordnetAnnotator(nlp.lang), after='tagger')
#nasari_df = pd.read_csv('./NASARIembed+UMBC_w2v.txt', header=None, sep=' ', skiprows=1)

def load_defs(path):
  with open(path, 'r') as file:
    lines = file.readlines()
  
  terms = []
  definitions = []  

  for line in lines:
    chunks = line.lower().strip().split('\t')
    terms.append(chunks[0])
    definitions.append(chunks[1:])
  return terms, definitions

def find_form(term, definitions):
  stopwords = nltk.corpus.stopwords.words('english')

  domains = []
  synsets = []
  context = []

  for definition in definitions:
    # 1. PoS tagging
    text = nlp(definition)
    #for token in text:
    #  print(f'{token}[{token.dep_}] ', end='', flush=True)
    #print()

    # 2. Estrarre SUBJ con relativi ADJ e OBJ
    #tags = ['nsubj', 'ROOT', 'dobj', 'pobj', 'conj', 'amod']
    tags = ['ROOT', 'ADJ']
    exclude_tags = '.,:;!?()”“…'
    subjs = filter(lambda token: token.dep_ in tags, text)
    relevant_words = filter(lambda token: token.text not in stopwords, text)
    relevant_words = filter(lambda token: token.text not in exclude_tags, relevant_words)
    relevant_words = list(relevant_words)
    #context.extend(relevant_words)
    #print(relevant_words)

    # 3. Ricavare SUBJ principale (es. più ricorrente, o iperonimo soggetti?)
    domains.extend(list(map(lambda t: t.text, subjs)))
    for token in relevant_words:
      #print(f'WordNet domains for {token}: ')
      #print(token._.wordnet.wordnet_domains())
      #print('Synets: ', token._.wordnet.synsets())
      #print()
      domains.extend(token._.wordnet.wordnet_domains())
      synsets.extend(token._.wordnet.synsets())
      context.append(token.text)

    # 4. Synset di iperonimo ricavato
    # 5. Per i figli dell'iperonimo ottenere gloss
    # 6. W.O. delle gloss con definizioni
    # 7. trovare il max
    #print()
  
  counter = Counter(domains)
  most_common = counter.most_common(10)
  print("---DOMAINS:", most_common)
  #print(most_common)

  #counter = Counter(synsets)
  #most_common = counter.most_common(10)
  #print(most_common)

  #print('---------- CONTEXT ----------')
  c_counter = Counter(context)
  c_common = c_counter.most_common(20)
  print("---CONTEXT:", c_common)
  print()

  best_synset = None
  best_score = 0

  context = list(map(lambda c: c[0], c_common))
  context = nlp(' '.join(context))

  for lemma in most_common:
    print(f'Exploring domain "{lemma[0]}"')
    synset, score = find_most_similar_synset(wn.synsets(lemma[0]),
                                             context,
                                             lower_bound=best_score,
                                             verbose=False)
    print(f'Best score: {score} with synset {synset}')
    if synset:
      print(synset.definition())
    print()
    if score > best_score:
      best_synset = synset
      best_score = score

  return best_synset
  return most_common[0]

def find_most_similar_hyponym(synset, context, level=0, lower_bound=0, verbose=False):
  definition = synset.definition()
  if verbose:
    print('-'*level + f'Level {level}, synset: {synset}')
    print('-'*level + f'Definition: {definition}')

  definition = nlp(definition)
  definition = nlp(' '.join([str(t) for t in definition if not t.is_stop]))

  best_hyponym = synset
  best_score = definition.similarity(context) #compute_overlap_sim(context, definition)
  if best_score <= lower_bound:
    return best_hyponym, best_score

  for hyponym in synset.hyponyms():
    syn, score = find_most_similar_hyponym(hyponym, context, level+1, best_score, verbose)
    if score > best_score:
      best_hyponym = syn
      best_score = score

  return best_hyponym, best_score

def find_most_similar_synset(synsets, context, lower_bound=0, verbose=False, recursive=True):
  best_synset = None
  best_score = 0

  for synset in synsets:
    synset, score = synset, 0
    if recursive:
      synset, score = find_most_similar_hyponym(synset, context, lower_bound=lower_bound, verbose=verbose)
    else:
      definition = nlp(synset.definition())
      definition = nlp(' '.join([str(t) for t in definition if not t.is_stop]))
      score = definition.similarity(context)

    if score > best_score:
      best_synset = synset
      best_score = score
  return best_synset, best_score

def find_form_vale_version(term, definitions):
  stopwords = nltk.corpus.stopwords.words('english')
  domains = []
  relevant_words = []
  subjects = []

  print(f'\nTermine target: {term}')

  for definition in definitions:
    # 1. PoS tagging
    text = nlp(definition)

    # 2. Estrarre SUBJ con relativi ADJ e OBJ
    tags = ['ROOT', 'dobj', 'pobj', 'amod']
    pod = ['NOUN', 'ADJ']
    filtered_text = filter(lambda token: token.dep_ in tags or token.dep_ in pod, text)

    for token in filtered_text:
      if token.dep_ == 'ROOT':
        subjects.append(token.text)
      else:
        relevant_words.append(token.text)
        # TODO: estende le parole importanti con il loro dominio
        # relevant_words.extend(filter(lambda tok: tok not in stopwords, token._.wordnet.wordnet_domains()))

  # 4. Synset di iperonimo ricavato ripetere finche punteggio new_hyper_score > old_hyper_score
  old_hyper_score = 0
  new_hyper_score = 0
  target_subject = collections.Counter(subjects).most_common(1)[0][0]
  print('Most common subject:', target_subject)

  if len(wn.synsets(target_subject)) > 0:
    synset_list = wn.synsets(target_subject)
    context = nlp(' '.join(relevant_words))
    subject_synset = find_most_similar_synset(synset_list, context, recursive=False)[0]
  else:
    # da valutare
    return target_subject

  depth = 1
  while new_hyper_score >= old_hyper_score:
    hyponyms = subject_synset.hyponyms()
    if len(hyponyms) == 0:
      break

    # 5. Per i figli dell'iperonimo ottenere gloss
    gloss_hyponyms = map(lambda hyp: (hyp, hyp.definition()), hyponyms)
    scores = []
    for couple in gloss_hyponyms:
      print(couple)
      # 6. W.O. tra contesto (relevant_words) e parole della gloss
      score = compute_overlap_sim(relevant_words, couple[1])
      scores.append(score)

    # 7. trovare il max e settare old e new score
    old_hyper_score = new_hyper_score
    new_hyper_score = max(scores)
    index_max = scores.index(new_hyper_score)
    subject_synset = hyponyms[index_max]
    depth += 1

  print('Risultato:', subject_synset, "Definizione: ", subject_synset.definition(), "Depth:", depth)
  return subject_synset, depth

def min_distance(w1, w2):
  s1 = nlp(w1)
  s2 = nlp(w2)

  print(s1)

def compute_min_distance_sim(context, text):
  return 0

nasari = {}
def load_nasari(path):
  global nasari
  with open('./Wiki_NASARI_unified_english.txt', 'r') as file:
    lines = file.readlines()
    
  for line in lines[1:]:
    id = line.split(' ')[0]
    v = list(map(lambda s: float(s), line.split(' ')[1:]))
    nasari[id] = v

def get_vector(bn_id):
  global nasari
  if bn_id not in nasari:
    print(f'{bn_id} not in NASARI')
    return [0]*300
  return nasari[bn_id]
  """global cache
  if bn_id in cache:
    return cache[bn_id]

  with open('./Wiki_NASARI_unified_english.txt', 'r') as file:
    for i, line in enumerate(file):
      if i == 0:
        continue
      id = line.split(' ')[0]
      if bn_id == id:
        res = list(map(lambda s: float(s), line.split(' ')[1:]))
        cache[bn_id] = res
        return res"""

def similarity(v1, v2):
  #TODO W.O instead of cosine similarity
  v1 = np.array(v1)
  v2 = np.array(v2)
  return (v1*v2).sum() / (np.sqrt(np.square(v1).sum())*np.sqrt(np.square(v2).sum()))

words = []
def compute_overlap_sim(context, text):
  stopwords = nltk.corpus.stopwords.words('english')
  #print(f'Computing similarity between {context} and {text}')
  global words
  words.extend(context)
  words.extend(map(lambda t: t.text, nlp(text)))

  tot_score = []
  tokens = list(filter(lambda t: t.text not in stopwords, nlp(text)))
  for word1 in context:
    token1 = nlp(word1)[0]
    for word2 in tokens:
      sim = token1.similarity(word2)
      tot_score.append(sim)
      #print(f'Scores for {word1} and {word2.text}: ', sim)
      continue
      s1 = babelnet_ids[word1] #babelnet_ids for word1
      s2 = babelnet_ids[word2.text] #babelnet_ids for word2

      max_score = 0

      print(f'Scores for {word1} and {word2.text}')
      for id1 in s1:
        for id2 in s2:
          v1 = get_vector(id1)
          v2 = get_vector(id2)

          sim = similarity(v1, v2)
          print(f'{id1}-{id2}: {sim}')
          if sim > max_score:
            max_score = sim
      tot_score.append(max_score)

  return sum(tot_score) / (len(context)+len(tokens))


if __name__ == '__main__':
  with open('babelnet_ids.json', 'r') as file:
    babelnet_ids = json.load(file)

  #load_nasari('Wiki_NASARI_unified_english.txt')

  nltk.download('punkt')
  nltk.download('averaged_perceptron_tagger')
  nltk.download('wordnet')

  terms, definitions = load_defs('esercitazione2.tsv')
  # print(f'Term: {terms[0]}, definition[0]: {definitions[0][0]}')
  
  forms = []
  #TODO: per ogni coppia termine/definizioni
  for t, d in zip(terms, definitions):
    forms.append(find_form_vale_version(t, d))

  words = set(words)
  with open('wordlist_babelnet.txt', 'w') as file:
    for w in words:
      file.write(w+'\n')
  
  
  # for definition in definitions:
    # forms.append(find_form_vale_version(definition))

  print('---------------- RESULTS -------------------')
  for term, form in zip(terms, forms):
    print(f'Ground: {term} - found: {form}')
