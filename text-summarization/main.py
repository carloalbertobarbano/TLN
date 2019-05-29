from typing import List, Dict
from pprint import pprint

import nltk
import numpy as np
from scipy.stats import rankdata
import argparse

def load_nasari(path):
  with open(path, 'r') as file:
    lines = file.readlines()

  lines = filter(lambda line: not line.startswith('#') and line.strip(), lines)
  lines = [line.split(';') for line in lines]
  nasari = {}

  for vec in lines:
    babel_id = vec[0]
    word = vec[1] #.lower()
    synsets = filter(lambda s: s and s.strip(), vec[2:])
    if word not in nasari:
      #print(f"Word {word} is already in nasari!")
      nasari[word] = {'babel_id': babel_id, 'synsets': []}

    nasari[word]['synsets'].append({})
    synsets = list(filter(lambda synset: '_' in synset, synsets))
    for synset in synsets:
      synset_name, synset_weight = synset.split('_')
      nasari[word]['synsets'][-1][synset_name] = float(synset_weight)

  return nasari

def load_text(path):
  with open(path, 'r') as file:
    lines = file.readlines()
  lines = list(map(lambda line: line.strip(), filter(lambda line: not line.startswith('#') and line.strip(), lines)))
  return lines

def create_context(words: List[str], nasari):
  context = [(w, nasari[w]) for w in words if w in nasari]
  return context

def remove_stopwords(words: List[str]):
  stopwords = nltk.corpus.stopwords.words('english')
  return list(filter(lambda word: word.replace('.', '') not in stopwords, words))

def rank_paragraphs_by_keywords(keywords: List[str], paragraphs: List[str]):
  return list(
    map(lambda paragraph: sum([paragraph.lower().count(key.lower()) for key in keywords]), paragraphs)
  )
def remove_punctuation(string):
  chars = '.,:;!?()”“…'
  for c in chars:
    string = string.replace(c, '')
  return string

def find_most_frequent_words(paragraph: str):
  words = remove_stopwords(set(remove_punctuation(paragraph).split(' ')))
  freqs = map(lambda w: (w, paragraph.lower().count(w.lower())), words)
  freqs = filter(lambda freq: len(freq[0]) > 3, freqs)

  return sorted(freqs, key=lambda f: f[1], reverse=True)#[:50]

def weighted_overlap(v1: Dict[str, float], v2: Dict[str, float]):
  intersection = set(v1.keys()).intersection(set(v2.keys()))

  rank1 = {}
  rank2 = {}

  for i, k in enumerate(v1.keys()):
    rank1[k] = rankdata(list(v1.values()))[i]
  
  for i, k in enumerate(v2.keys()):
    rank2[k] = rankdata(list(v2.values()))[i]
  
  sum = 0
  div = 0.001
  for i, dim in enumerate(intersection):
    sum += 1./(rank1[dim] + rank2[dim])
    div += 1./(2*(i+1))

  return sum/div, intersection

def similarity(w1, w2):
  max = 0
  intersect = 0

  for v1 in w1[1]['synsets']:
    for v2 in w2[1]['synsets']: 
      wo, inter = weighted_overlap(v1, v2)
      wo = np.sqrt(wo)
      if wo > max:
        max = wo
        intersect = inter
  
  if max > 0:
    print(f'WO between {w1[0]} and {w2[0]}: {max} (intersection: {intersect})')
  return max

def summarize_text(paragraphs: List[str], nasari, compression=10):
  title, body = paragraphs[0], paragraphs[1:]

  #Keywords from title
  print("Using title for keywords:", title)
  keywords = remove_stopwords(title.split(' '))
  print("Filtered keywords (no stopwords):", keywords)

  ranks = rank_paragraphs_by_keywords(keywords, body)
  print("Ranks:", ranks) 
  print('Best paragraph:', np.argmax(ranks))
  
  freq_words = find_most_frequent_words(' '.join(body))
  print(f'Most common words in body:', freq_words)

  context = create_context(np.array(freq_words)[:, 0], nasari)
  pprint(context)
  print("Words in context:", np.array(context)[:, 0])

  for i in range(len(context)):
    for j in range(len(context)):
      if i != j:
        w1 = context[i]
        w2 = context[j]
        wo = similarity(w1, w2)
        #if wo > 0:
        #  print(f'WO between {w1[0]} and {w2[0]}:', weighted_overlap(w1[1]['synsets'], w2[1]['synsets']))
  return paragraphs

parser = argparse.ArgumentParser()
parser.add_argument('-i', help='input file path', type=str)
parser.add_argument('-o', help='output path', type=str)
parser.add_argument('-c', help='compression rate', type=int, default=10)
parser.add_argument('-n', help='nasari path', type=str, default='./dd-small-nasari-15.txt')

if __name__ == '__main__':
  args = parser.parse_args()

  nasari = load_nasari(args.n)
  #print("NASARI[power]:", nasari['power'])
  text = load_text(args.i)
  summarized_text = summarize_text(text, nasari, args.c)

  with open(args.o, 'w') as file:
    for paragraph in summarized_text:
      file.write(paragraph)
      file.write('\n')