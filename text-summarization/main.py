from typing import List

import nltk
import numpy as np
import argparse

def load_nasari(path):
  with open(path, 'r') as file:
    lines = file.readlines()

  lines = filter(lambda line: not line.startswith('#') and line.strip(), lines)
  lines = [line.split(';') for line in lines]
  nasari = {}

  for vec in lines:
    babel_id = vec[0]
    word = vec[1]
    synsets = filter(lambda s: s and s.strip(), vec[2:])
    nasari[word] = {'babel_id': babel_id, 'synsets': {}}

    synsets = list(filter(lambda synset: '_' in synset, synsets))
    for synset in synsets:
      synset_name, synset_weight = synset.split('_')
      nasari[word]['synsets'][synset_name] = float(synset_weight)

  return nasari

def load_text(path):
  with open(path, 'r') as file:
    lines = file.readlines()
  lines = list(map(lambda line: line.strip(), filter(lambda line: not line.startswith('#') and line.strip(), lines)))
  return lines

def create_context(paragraphs: List[str], nasari):
  context = []
  for paragraph in paragraphs:
    context.append([nasari[w] for w in paragraph if w in nasari])
  return context

def rank_paragraphs_by_keywords(keywords: List[str], paragraphs: List[str]):
  return list(map(lambda paragraph: sum([paragraph.lower().count(key.lower()) for key in keywords]), paragraphs))

def remove_stopwords(words: List[str]):
  stopwords = nltk.corpus.stopwords.words('english')
  return list(filter(lambda word: word.lower().replace('.', '') not in stopwords, words))

def find_most_frequent_words(paragraph: str):
  words = remove_stopwords(set(paragraph.lower().replace('.', '').split(' ')))
  freqs = map(lambda w: (w, paragraph.lower().count(w)), words)
  return sorted(freqs, key=lambda f: f[1], reverse=True)[:10]

def summarize_text(paragraphs: List[str], compression=10):
  title, body = paragraphs[0], paragraphs[1:]

  #Keywords from title
  print("Using title for keywords:", title)
  keywords = remove_stopwords(title.split(' '))
  print("Filtered keywords (no stopwords):", keywords)

  ranks = rank_paragraphs_by_keywords(keywords, body)
  print("Ranks:", ranks) 
  print('Best paragraph:', np.argmax(ranks))
  
  freq_words = find_most_frequent_words(body[np.argmax(ranks)])
  print(f'Most common words in paragraph {np.argmax(ranks)}:', freq_words)

  return paragraphs

parser = argparse.ArgumentParser()
parser.add_argument('-i', help='input file path', type=str)
parser.add_argument('-o', help='output path', type=str)
parser.add_argument('-c', help='compression rate', type=int, default=10)

if __name__ == '__main__':
  args = parser.parse_args()

  nasari = load_nasari('./dd-small-nasari-15.txt')
  text = load_text(args.i)
  summarized_text = summarize_text(text, args.c)

  with open(args.o, 'w') as file:
    for paragraph in summarized_text:
      file.write(paragraph)
      file.write('\n')