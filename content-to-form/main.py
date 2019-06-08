import nltk
import spacy
import spacy_wordnet

from spacy_wordnet.wordnet_annotator import WordnetAnnotator
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

  for definition in definitions:
    # 1. PoS tagging
    text = nlp(definition)
    for token in text:
      print(f'{token}[{token.dep_}] ', end='', flush=True)
    print()

    # 2. Estrarre SUBJ con relativi ADJ e OBJ
    tags = ['nsubj', 'ROOT', 'dobj', 'pobj', 'conj']
    relevant_words = filter(lambda token: token.dep_ in tags, text)
    relevant_words = filter(lambda token: token.text not in stopwords, relevant_words)
    relevant_words = list(relevant_words)
    #print(relevant_words)

    # 3. Ricavare SUBJ principale (es. più ricorrente, o iperonimo soggetti?)
    for token in relevant_words:
      print(f'WordNet domains for {token}: ')
      print(token._.wordnet.wordnet_domains())
      print()

    # 4. Synset di iperonimo ricavato
    # 5. Per i figli dell'iperonimo ottenere gloss
    # 6. W.O. delle gloss con definizioni
    # 7. trovare il max
    print()
  pass

if __name__ == '__main__':
  nltk.download('punkt')
  nltk.download('averaged_perceptron_tagger')
  nltk.download('wordnet')

  terms, definitions = load_defs('esercitazione2.tsv')
  print(f'Term: {terms[0]}, definition[0]: {definitions[0][0]}')

  find_form(definitions[0])