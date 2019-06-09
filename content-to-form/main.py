import nltk
import spacy
import spacy_wordnet
import collections
#import pandas as pd

from spacy_wordnet.wordnet_annotator import WordnetAnnotator
from collections import Counter

from nltk.corpus import wordnet as wn

nlp = spacy.load("en")
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

def find_form(definitions):
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
    tags = ['nsubj', 'ROOT', 'dobj', 'pobj', 'conj', 'amod']
    exclude_tags = '.,:;!?()”“…'
    #relevant_words = filter(lambda token: token.dep_ in tags, text)
    relevant_words = filter(lambda token: token.text not in stopwords, text)
    relevant_words = filter(lambda token: token.text not in exclude_tags, relevant_words)
    relevant_words = list(relevant_words)
    #context.extend(relevant_words)
    #print(relevant_words)

    # 3. Ricavare SUBJ principale (es. più ricorrente, o iperonimo soggetti?)
    
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
  #print(most_common)

  counter = Counter(synsets)
  most_common = counter.most_common(10)
  #print(most_common)

  print('---------- CONTEXT ----------')
  c_counter = Counter(context)
  c_common = c_counter.most_common(10)
  print(c_common)
  print()

  return most_common[0]

def find_form_vale_version(term, definitions):
  stopwords = nltk.corpus.stopwords.words('english')
  domains = []
  relevant_words = []
  subjects = []

  print(f'Termine target: {term}\n')

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
        # relevant_words.extend(token._.wordnet.wordnet_domains())


  # 4. Synset di iperonimo ricavato ripetere finche punteggio new_hyper_score > old_hyper_score
  old_hyper_score = 0
  new_hyper_score = 0
  target_subject = collections.Counter(subjects).most_common(1)[0][0]
  print('Most common subject:', target_subject)

  if len(wn.synsets(target_subject)) > 0:
    subject_synset = wn.synsets(target_subject)[0]
  else:
    return None

  while new_hyper_score >= old_hyper_score:
    # TODO: forse considerare anche gli altri?
    print(f'\nSoggetto: {subject_synset}')
    hyponyms = subject_synset.hyponyms()
    print(f'Hyponyms of {subject_synset}:', hyponyms)
    
    ## SENZA BREAK SI ROMPE ##
    if len(hyponyms) == 0:
      break

    # 5. Per i figli dell'iperonimo ottenere gloss
    gloss_hyponyms = map(lambda hyp: (hyp, hyp.definition()), hyponyms)
    scores = []
    for couple in gloss_hyponyms:
      print(couple)
      # 6. W.O. tra contesto (relevant_words) e parole della gloss
      # TODO: da implementare/importare
      score = compute_overlap_sim(relevant_words, couple[1])
      scores.append(score)

    # 7. trovare il max e settare old e new score
    old_hyper_score = new_hyper_score
    new_hyper_score = max(scores)
    index_max = scores.index(new_hyper_score)
    subject_synset = hyponyms[index_max]
  
  print('Risultato:', subject_synset)

def min_distance(w1, w2):
  s1 = nlp(w1)
  s2 = nlp(w2)

  print(s1)

def compute_min_distance_sim(context, text):
  return 0

def get_vector(synset):
  with open('./NASARIembed+UMBC_w2v.txt', 'r') as file:
    for i, line in enumerate(file):
      if i == 0:
        continue
      id = line.split(' ')[0]
      if synset == id:
        return list(map(lambda s: float(s), line.split(' ')[1:]))

words = []
def compute_overlap_sim(context, text):
  global words
  words.extend(context)
  words.extend(map(lambda t: t.text, nlp(text)))
  return 0


if __name__ == '__main__':
  nltk.download('punkt')
  nltk.download('averaged_perceptron_tagger')
  nltk.download('wordnet')

  terms, definitions = load_defs('esercitazione2.tsv')
  # print(f'Term: {terms[0]}, definition[0]: {definitions[0][0]}')
  
  #TODO: per ogni coppia termine/definizioni
  for t, d in zip(terms, definitions):
    find_form_vale_version(t, d)
  words = set(words)
  with open('wordlist_babelnet.txt', 'w') as file:
    for w in words:
      file.write(w+'\n')
  
  # forms = []
  # for definition in definitions:
    # forms.append(find_form_vale_version(definition))

  # print('----------------')
  # for term, form in zip(terms, forms):
    # print(f'Ground: {term} - found: {form[0]}')
