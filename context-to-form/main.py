import nltk

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
  for definition in definitions:
    # 1. PoS tagging
    text = nltk.word_tokenize(definition)
    tags = nltk.pos_tag(text)
    print(tags)

    # 2. Estrarre SUBJ con relativi ADJ e OBJ
    # 3. Ricavare SUBJ principale (es. più ricorrente, o iperonimo soggetti?)
    # 4. Synset di iperonimo ricavato
    # 5. Per i figli dell'iperonimo ottenere gloss
    # 6. W.O. delle gloss con definizioni
    # 7. trovare il max
  pass

if __name__ == '__main__':
  nltk.download('punkt')
  nltk.download('averaged_perceptron_tagger')

  terms, definitions = load_defs('esercitazione2.tsv')
  print(f'Term: {terms[0]}, definition[0]: {definitions[0][0]}')

  find_form(definitions[0])