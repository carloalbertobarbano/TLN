import itertools
import numpy as np
from typing import List
from pprint import pprint

import requests
import json
import time

def parse_senses2synsets(path):
  terms = {}
  with open(path, 'r') as file:
    lines = file.readlines()
  
  curr_term = None
  for line in lines:
    if line.startswith('#'):
      curr_term = line.replace('#', '').strip().lower()
      terms[curr_term] = []
    else:
      terms[curr_term].append(line.strip())
  return terms

def parse_nasari(path):
  with open(path, 'r') as file:
    lines = file.readlines()
  nasari = {}

  for line in lines:
    terms = line.split('\t')
    bn_id = terms[0].split('__')[0]
    name = terms[0].split('__')[1].replace('_', ' ')
    vector = np.array(list(map(lambda t: float(t), terms[1:])))
    nasari[bn_id] = {'name': name, 'v': vector}
  return nasari

def cos_similarity(v1, v2):
  return (v1*v2).sum() / (np.sqrt(np.square(v1).sum())*np.sqrt(np.square(v2).sum()))

def find_max_similarity(synsets1: List[str], synsets2: List[str], nasari):
  max_sim, _s1, _s2 = 0, None, None

  for s1, s2 in itertools.product(synsets1, synsets2):
    if s1 not in nasari or s2 not in nasari:
      continue
    nasari1 = nasari[s1]
    nasari2 = nasari[s2]

    #print(f"Nasari for {s1}: {nasari1['v'][:10]}")
    #print(f"Nasari for {s2}: {nasari2['v'][:10]}")

    similarity = cos_similarity(nasari1['v'], nasari2['v'])
    #print(f"Similarity: {similarity}")

    if similarity > max_sim:
      max_sim = similarity
      _s1 = s1
      _s2 = s2
  return max_sim, _s1, _s2

def parse_terms(path):
  with open(path, 'r') as file:
    lines = file.readlines()
  
  pairs = []
  score = []

  for line in lines:
    terms = line.split('\t')
    #print(terms)
    pairs.append((terms[0].lower().strip(), terms[1].lower().strip()))
    score.append(float(terms[2].strip()))
  
  return pairs, score

def get_gloss(synset):
  r = requests.get('https://babelnet.io/v5/getSynset?id={}&key=558130e3-cf9a-4501-86ce-89ae96041923'.format(synset))
  response = r.json()
  
  if 'glosses' in response:
    return response['glosses'][0]['gloss']
  return 'NONE'

USE_BABELNET = False

if __name__ == '__main__':
  #print(cos_similarity(np.array([1,2,3,4]), np.array([1,2,3,4])))
  #exit(0)

  words, hand_score = parse_terms('./extracted2.it.test.data.txt')
  senses2synsets = parse_senses2synsets('./SemEval17_IT_senses2synsets.txt')
  nasari = parse_nasari('./mini_NASARI.tsv')

  similarity = []
  synsets = []

  glosses = {}
  if not USE_BABELNET:
    with open('glosses.json', 'r') as file:
      glosses = json.load(file)

  for i, (term1, term2) in enumerate(words):
    synsets1 = senses2synsets[term1]
    synsets2 = senses2synsets[term2]

    print(f"-------- Terms: w1={term1} w2={term2}")
    
    sim, s1, s2 = find_max_similarity(synsets1, synsets2, nasari)
    sim += 1
    sim *= 2

    if USE_BABELNET:
      glosses[s1] = get_gloss(s1)
      glosses[s2] = get_gloss(s2)

    if s1 not in glosses:
      glosses[s1] = 'NONE'
    if s2 not in glosses:
      glosses[s2] = 'NONE'

    print(f'\tSense for w1: {s1} - Gloss: {glosses[s1]}')
    print(f'\tSense for w2: {s2} - Gloss: {glosses[s2]}')
    print(f'\tSimilarity: {sim:.2f}')
    print(f'\tHand-score: {hand_score[i]}')
    print()
    similarity.append(sim)
    synsets.append((s1, s2))

    if USE_BABELNET:
      time.sleep(1)
      

  if USE_BABELNET:
    with open('glosses.json', 'w') as file:
      json.dump(glosses, file)
  
  similarity = np.array(similarity)

  print("Correlation between hand_score and similarity: ", np.corrcoef(hand_score, similarity)[0, 1])  
