import nltk
import spacy
import collections
import warnings

warnings.filterwarnings('ignore')
from spacy_wordnet.wordnet_annotator import WordnetAnnotator
from collections import Counter

from nltk.corpus import wordnet as wn

import sys

nlp = spacy.load("en_core_web_lg")
nlp.add_pipe(WordnetAnnotator(nlp.lang), after='tagger')


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
  print()
  print('-'*25)
  print(f'Termine target: {term}')
  stopwords = nltk.corpus.stopwords.words('english')

  domains = []
  synsets = []
  context = []

  for definition in definitions:
    # 1. PoS tagging
    text = nlp(definition)

    # 2. Estrarre soggetti (ROOT) con relativi ADJ
    tags = ['ROOT', 'ADJ']
    subjs = filter(lambda token: token.dep_ in tags, text)

    # 3. Termini rilevanti = tutti i termini della definizione - stopwords
    exclude_tags = '.,:;!?()”“…'
    relevant_words = filter(lambda token: token.text not in stopwords, text)
    relevant_words = filter(lambda token: token.text not in exclude_tags, relevant_words)
    relevant_words = list(relevant_words)

    # 4. Dominio = soggetti + wordnet domains del contesto
    #   Synsets = synsets del contesto
    #   Context = termini rilevanti
    domains.extend(list(map(lambda t: t.text, subjs)))
    for token in relevant_words:
      domains.extend(token._.wordnet.wordnet_domains())
      synsets.extend(token._.wordnet.synsets())
      context.append(token.text)

  # 5. Domini più frequenti (top 10)
  counter = Counter(domains)
  most_common = counter.most_common(10)
  print("---DOMAINS:", most_common)

  # 6. Context = termini del contesto più frequenti
  c_counter = Counter(context)
  c_common = c_counter.most_common(15)
  print("---CONTEXT:", c_common)
  print()
  context = list(map(lambda c: c[0], c_common))
  context = nlp(' '.join(context))

  best_synset = None
  best_score = 0
  best_level = 0

  # 7. Esplorazione dei domini -> top down
  #   Per ogni synset di un dominio, cerco tra gli iponimi quello che
  #   ha score di similarità maggiore rispetto al contesto
  #   (idea: da termine generale a termine specifico)
  for lemma in most_common:
    print(f'Exploring domain "{lemma[0]}"')
    synset, score, level = find_most_similar_synset(wn.synsets(lemma[0]),
                                                    context,
                                                    lower_bound=best_score - 0.5,
                                                    verbose=False)
    print(f'Best score: {score} with synset {synset}')
    if synset:
      print(synset.definition())
    print()
    if score > best_score:
      best_synset = synset
      best_score = score
      best_level = level

  return best_synset, best_level


def find_most_similar_hyponym(synset, context, level=1, lower_bound=0, verbose=False):
  definition = synset.definition()
  if verbose:
    print('-' * level + f'Level {level}, synset: {synset}')
    print('-' * level + f'Definition: {definition}')

  definition = nlp(definition)
  definition = nlp(' '.join([str(t) for t in definition if not t.is_stop]))

  best_hyponym = synset
  best_score = definition.similarity(context)
  best_level = level
  if best_score < lower_bound:
    return best_hyponym, best_score, best_level - 1

  for hyponym in synset.hyponyms():
    syn, score, c_level = find_most_similar_hyponym(hyponym, context, level + 1, best_score, verbose)
    if score > best_score:
      print(f'\t{c_level}: {syn}')
      best_hyponym = syn
      best_score = score
      best_level = c_level

  return best_hyponym, best_score, best_level


def find_most_similar_synset(synsets, context, lower_bound=0, verbose=False, recursive=True):
  best_synset = None
  best_score = 0
  level = 1

  for synset in synsets:
    synset, score, c_level = synset, 0, level
    if recursive:
      synset, score, c_level = find_most_similar_hyponym(synset, context, lower_bound=lower_bound, verbose=verbose)
    else:
      definition = nlp(synset.definition())
      definition = nlp(' '.join([str(t) for t in definition if not t.is_stop]))
      score = definition.similarity(context)

    if score > best_score:
      best_synset = synset
      best_score = score
      level = c_level
  return best_synset, best_score, level


def find_form_v2(term, definitions):
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

  # 4. Synset di iperonimo ricavato ripetere finche punteggio new_hyper_score > old_hyper_score
  old_hyper_score = 0
  new_hyper_score = 0
  target_subject = collections.Counter(subjects).most_common(1)[0][0]
  subject_synset = None
  print('Most common subject:', target_subject)

  if len(wn.synsets(target_subject)) > 0:
    synset_list = wn.synsets(target_subject)
    context = nlp(' '.join(relevant_words))
    subject_synset = find_most_similar_synset(synset_list, context, recursive=False)[0]
  else:
    # da valutare
    return target_subject

  depth = 1
  while subject_synset and new_hyper_score >= old_hyper_score:
    hyponyms = subject_synset.hyponyms()
    if len(hyponyms) == 0:
      break

    # 5. Per i figli dell'iperonimo ottenere gloss
    gloss_hyponyms = map(lambda hyp: (hyp, hyp.definition()), hyponyms)
    scores = []
    for couple in gloss_hyponyms:
      print(couple)
      # 6.similarity tra contesto (relevant_words) e parole della gloss
      score = compute_similarity(relevant_words, couple[1])
      scores.append(score)

    # 7. trovare il max e settare old e new score
    old_hyper_score = new_hyper_score
    new_hyper_score = max(scores)
    index_max = scores.index(new_hyper_score)
    subject_synset = hyponyms[index_max]
    depth += 1

  print('Risultato:', subject_synset, "Definizione: ", subject_synset.definition(), "Depth:", depth, "\n")
  return subject_synset, depth


def compute_similarity(context, text):
  stopwords = nltk.corpus.stopwords.words('english')
  # print(f'Computing similarity between {context} and {text}')

  tot_score = []
  tokens = list(filter(lambda t: t.text not in stopwords, nlp(text)))
  for word1 in context:
    token1 = nlp(word1)[0]
    for word2 in tokens:
      sim = token1.similarity(word2)
      tot_score.append(sim)

  return sum(tot_score) / (len(context) + len(tokens))


def sentence_similarity(s1, s2):
  s1 = nlp(s1)
  s1 = nlp(' '.join([str(t) for t in s1 if not t.is_stop]))

  s2 = nlp(s2)
  s2 = nlp(' '.join([str(t) for t in s2 if not t.is_stop]))

  return s1.similarity(s2)


def select_best_synset_by_defs(synsets, definitions):
  best_synset = None
  best_score = 0

  for synset in synsets:
    curr_score = 0
    for definition in definitions:
      curr_score += sentence_similarity(synset.definition(), definition)

    if curr_score > best_score:
      best_synset = synset
      best_score = curr_score
  return best_synset


if __name__ == '__main__':
  nltk.download('punkt')
  nltk.download('averaged_perceptron_tagger')
  nltk.download('wordnet')

  terms, definitions = load_defs('esercitazione2.tsv')
  if len(sys.argv) > 1:
    index = int(sys.argv[1])
    terms = [terms[index]]
    definitions = [definitions[index]]

  forms1 = []
  forms2 = []

  for t, d in zip(terms, definitions):
    forms1.append(find_form(t, d))
    forms2.append(find_form_v2(t, d))

  print('---------------- RESULTS -------------------')
  for i, (form1, form2) in enumerate(zip(forms1, forms2)):
    synset1, depth1 = form1
    synset2, depth2 = form2
    depth1 = synset1.min_depth()
    depth2 = synset2.min_depth()
    best_synset = synset1

    # Per scegliere la form migliore si preferisce synset con depth più alta (= più specifico)
    if depth2 > depth1:
      best_synset = synset2

    # A parità di depth si seleziona synset la cui definizione ha score di similarità maggiore
    # con le definizioni date
    if depth1 == depth2:
      best_synset = select_best_synset_by_defs((synset1, synset2), definitions[i])

    print(f'Ground: {terms[i]}')
    print(f'Found: {[str(x.name()) for x in best_synset.lemmas()]}')
    print(f'(synsets) [s1: {(synset1, depth1)} s2: {(synset2, depth2)}]')
    print('-'*25)
    print()
