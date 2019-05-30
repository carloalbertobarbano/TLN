from typing import List, Dict
from pprint import pprint

import nltk
import numpy as np
from scipy.stats import rankdata
import argparse

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

def load_nasari(path, case_sensitive=True):
  with open(path, 'r') as file:
    lines = file.readlines()

  lines = filter(lambda line: not line.startswith('#') and line.strip(), lines)
  lines = [line.split(';') for line in lines]
  nasari = {}

  for vec in lines:
    babel_id = vec[0]
    word = vec[1]
    if not case_sensitive:
      word = word.lower()

    synsets = filter(lambda s: s and s.strip(), vec[2:])
    if word not in nasari:
      nasari[word] = []

    nasari[word].append({'babel_id': babel_id, 'synsets': {}})
    synsets = list(filter(lambda synset: '_' in synset, synsets))
    for synset in synsets:
      synset_name, synset_weight = synset.split('_')
      nasari[word][-1]['synsets'][synset_name] = float(synset_weight)

  return nasari

def load_text(path):
  with open(path, 'r') as file:
    lines = file.readlines()
  lines = list(map(
    lambda line: line.strip(), 
    filter(lambda line: not line.startswith('#') and line.strip(), lines)
  ))
  return lines

def create_context(words: List[str], nasari, case_sensitive=True):
  if case_sensitive:
   return [(w, nasari[w.capitalize()]) for w in words if w.capitalize() in nasari][:10]
  else:
   return [(w.lower(), nasari[w.lower()]) for w in words if w.lower() in nasari][:10] 

def remove_stopwords(words: List[str]):
  stopwords = nltk.corpus.stopwords.words('english')
  return list(filter(lambda word: word.replace('.', '').lower() not in stopwords, words))

def rank_paragraphs_by_keywords(keywords: List[str], paragraphs: List[str]):
  return list(map(
    lambda paragraph: sum([paragraph.lower().count(key.lower()) for key in keywords]), 
    paragraphs
  ))

def remove_punctuation(string):
  chars = '.,:;!?()”“…'
  for c in chars:
    string = string.replace(c, '')
  string = string.replace("’s", '')
  string = string.replace("-", " ")
  return string

def find_most_frequent_words(paragraph: str):
  words = remove_stopwords(set(map(lambda w: w.strip(), remove_punctuation(paragraph).lower().split(' '))))
  words = sorted(words)
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

  for v1 in w1[1]: #['synsets']:
    for v2 in w2[1]: #['synsets']: 
      wo, inter = weighted_overlap(v1['synsets'], v2['synsets'])
      wo = np.sqrt(wo)
      if wo > max:
        max = wo
        intersect = inter
  
  #if max > 0:
  #  print(f'WO between {w1[0]} and {w2[0]}: {max} (intersection: {intersect})')
  return max

def rank_paragraphs_by_wo(context, paragraphs: List[str], nasari, case_sensitive):
  ranks = []
  for paragraph in paragraphs:
    rank = 0
    for w2 in paragraph.split(' '):
      if case_sensitive:
        w2 = w2.capitalize()

      if w2 in nasari:
        rank += sum([similarity(w1, (w2, nasari[w2])) for w1 in context])
    ranks.append(rank)
  return ranks

def word_count(paragraphs: List[str]):
  return sum([len(p.split(' ')) for p in paragraphs])

def rank_paragraphs_by_cohesion(words_to_exclude: List[str], paragraphs: List[str]):
  # Count co-occurences of names (i.e. people, cities) excluding words common to all paragraphs
  occurences = []
  ranks = [0] * len(paragraphs)

  words_to_exclude = list(map(lambda w: w.lower(), words_to_exclude))
  #print("Words to exclude:", words_to_exclude)

  for paragraph in paragraphs:
    words = remove_punctuation(paragraph).split(' ')
    names = filter(
      lambda word: word[0].isupper() and word.lower() not in words_to_exclude and len(word) > 3, 
      words
    )
    names = remove_stopwords(names)
    occurences.append(names)
  
  #print("Relevant words in paragraphs:")
  #pprint(occurences)

  for i in range(len(paragraphs)-1):
    #for j in range(len(paragraphs)):
    j = i+1
    #for j in range(len(paragraphs)):
    #  if i != j:
    intersection = set(occurences[i]).intersection(set(occurences[j]))
    #print(f"Intersection between [{i}] and [{j}]:", intersection)
    ranks[i] = len(intersection) #max(ranks[i], len(intersection))
    #if j == len(paragraph)-1:
    ranks[j] += len(intersection)
  return ranks

def summarize_text(paragraphs: List[str], nasari, compression=10, ranking='wowo', case_sensitive=True):
  title, body = paragraphs[0], paragraphs[1:]

  num_words = word_count(body)
  target_num_words = num_words - num_words*compression/100.

  print(f'Total number of words: {num_words}')
  print(f'Target number of words: {target_num_words}')

  i = 0
  while num_words > target_num_words:  
    print(f'\n\nIteration {i}')
    print(f'Number of paragraphs: {len(body)}')
    freq_words = find_most_frequent_words(' '.join(body))
    #print(f'Most common words in body (top 10):')
    #pprint(freq_words[:10])

    context = create_context(np.array(freq_words)[:, 0], nasari, case_sensitive)
    #pprint(context)
    print("Words in context:")
    pprint(np.array(context)[:, 0])

    #Rank paragraph with Weighted Overlap with context
    wo_ranks = rank_paragraphs_by_wo(context, body, nasari, case_sensitive)
    wo_ranks = np.array(wo_ranks)

    #Keywords from title
    keywords = remove_stopwords(title.split(' '))

    keyword_ranks = []

    if ranking == 'wowo':
      print('Ranking paragraphs by WO with title. Context from title:')
      title_context = create_context(keywords, nasari, case_sensitive)
      pprint(np.array(title_context)[:, 0])
      keyword_ranks = rank_paragraphs_by_wo(title_context, body, nasari, case_sensitive)
    elif ranking == 'wokw':

      print("Ranking paragraphs by # of keywords in title:", title)
      print("Filtered keywords (no stopwords):")
      pprint(keywords)
      keyword_ranks = rank_paragraphs_by_keywords(keywords, body)
    else:
      print(f'Uknown ranking method {ranking}')
      exit(1)
    keyword_ranks = np.array(keyword_ranks)

    #Rank paragraphs by cohesion
    cohesion_ranks = np.array(rank_paragraphs_by_cohesion(keywords, body))

    print("WO ranks for paragraphs: ", wo_ranks)
    print("Title ranks:", keyword_ranks)

    tot_ranks = wo_ranks + keyword_ranks
    print('Combined ranks: ', tot_ranks)

    print("Cohesion ranks: ", cohesion_ranks)
    tot_ranks += cohesion_ranks
    print("Total ranks:", tot_ranks)

    min_paragraphs = np.where(cohesion_ranks == cohesion_ranks.min())
    print('Lowest paragraphs based on cohesion: ', min_paragraphs)
    print('Worst paragraph: ', np.argmin(tot_ranks))

    #print('Best paragraph:', np.argmax(tot_ranks))
    #print('Lowest paragraph:', np.argmin(tot_ranks))

    #Drop paragraph with lowest combined score
    index = np.argmin(tot_ranks)
    print("Removing paragraph", body[index][:40] + "..")
    del body[index]
    num_words = word_count(body)
    i += 1

  return [title] + body

parser = argparse.ArgumentParser()
parser.add_argument('-i', help='input file path', type=str)
parser.add_argument('-o', help='output path', type=str)
parser.add_argument('-c', help='compression rate', type=int, default=10)
parser.add_argument('-n', help='nasari path', type=str, default='./dd-small-nasari-15.txt')
parser.add_argument('--ranking', help='wowo or wokw', type=str, default='wokw')
parser.add_argument('--case_insensitive', help='load nasari as case insensitive', action="store_true", dest="case_insensitive", default=False)

if __name__ == '__main__':
  args = parser.parse_args()

  nasari = load_nasari(args.n, not args.case_insensitive)
  #print("NASARI[power]:", nasari['power'])
  text = load_text(args.i)
  summarized_text = summarize_text(text, nasari, args.c, args.ranking, not args.case_insensitive)

  with open(args.o, 'w') as file:
    for paragraph in summarized_text:
      file.write(paragraph)
      file.write('\n\n')