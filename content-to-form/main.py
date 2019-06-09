import nltk
import spacy
import spacy_wordnet

from spacy_wordnet.wordnet_annotator import WordnetAnnotator
from collections import Counter
#from nltk.corpus import wordnet

nlp = spacy.load('en')
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

if __name__ == '__main__':
  nltk.download('punkt')
  nltk.download('averaged_perceptron_tagger')
  nltk.download('wordnet')

  terms, definitions = load_defs('esercitazione2.tsv')
  print(f'Term: {terms[0]}, definition[0]: {definitions[0][0]}')

  forms = []
  for definition in definitions:
    forms.append(find_form(definition))
    print()
  
  print('----------------')
  for term, form in zip(terms, forms):
    print(f'Ground: {term} - found: {form[0]}')