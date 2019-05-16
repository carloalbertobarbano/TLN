import itertools
import numpy as np
from typing import List
from pprint import pprint
from nltk.corpus import wordnet
from nltk.corpus.reader.wordnet import Synset

WORDNET_MAX_DEPTH = 19

def load_csv(path_csv):
  with open(path_csv, 'r') as file:
    lines = file.readlines()
  lines = lines[1:]

  term1, term2, similarity = zip(*map(lambda l: l.split(','), lines))
  return term1, term2, np.array([float(s.strip()) for s in similarity])

def rec_hypernyms(s: Synset) -> List[Synset]:
  hypernyms = s.hypernyms()+s.instance_hypernyms()
  for hypernym in hypernyms:
    hypernyms.extend(get_hypernyms(hypernym))
  return hypernyms

def get_hypernyms(s: Synset) -> List[Synset]:
  if not s._all_hypernyms:
    #s._all_hypernyms = rec_hypernyms(s)
    s._all_hypernyms = []
    for synsets in s._iter_hypernym_lists():
      for synset in synsets:
        s._all_hypernyms.append(synset)
  return s._all_hypernyms

def common_hypernyms(hp1, hp2) -> List[Synset]:
  return list(set(hp1).intersection(set(hp2)))

diff_lcs = 0

def find_LCS(s1: Synset, s2: Synset) -> Synset:
  global diff_lcs
  g_lcs = s1.lowest_common_hypernyms(s2)
  #print(f"nltk lcs({s1},{s2})={g_lcs}")
  #s1._all_hypernyms = None
  #s2._all_hypernyms = None

  hp1 = get_hypernyms(s1)
  hp2 = get_hypernyms(s2)
  common_hps = common_hypernyms(hp1, hp2)
  lcs = sorted(common_hps, key=lambda hp: hp.max_depth(), reverse=True)

  if len(lcs) == 0:
    return None

  if lcs[0] != g_lcs[0]:
    print('LCS differ: ', lcs, g_lcs)
    diff_lcs += 1

  #print(f"custom lcs({s1},{s2})={lcs[0]}")
  return lcs[0]

def distance_to_hypernym(s: Synset, hp: Synset, min_distance=np.inf, curr_distance=0):
  if s == hp:
    return curr_distance

  for hypernym in s.hypernyms()+s.instance_hypernyms():
    if hypernym == hp and curr_distance+1 < min_distance:
      min_distance = curr_distance + 1
    
    min_distance = min(
      min_distance, 
      distance_to_hypernym(hypernym, hp, min_distance, curr_distance+1)
    )
  
  return min_distance

wrong_distances = 0

def synsets_distance(s1: Synset, s2: Synset):
  global wrong_distances

  lcs = find_LCS(s1, s2)
  if not lcs:
    return None
  
  dist1 = distance_to_hypernym(s1, lcs)
  dist2 = distance_to_hypernym(s2, lcs)

  custom_dist = dist1 + dist2
  nltk_dist = s1.shortest_path_distance(s2)

  if custom_dist != nltk_dist:
    wrong_distances += 1

    print()
    print(f"len({s1},{s2}) = Custom dist: {custom_dist}, nltk dist: {nltk_dist}")
    print(f"dist(s1={s1}, lcs={lcs}) = {dist1}, nltk(s1={s1}, lcs={lcs}) = {s1.shortest_path_distance(lcs)}")
    print(f"dist(s2={s2}, lcs={lcs}) = {dist2}, nltk(s2={s2}, lcs={lcs}) = {s2.shortest_path_distance(lcs)}")

    if dist1 != s1.shortest_path_distance(lcs):
      pprint(s1.tree(lambda s:s.hypernyms()+s.instance_hypernyms()))

    if dist2 != s2.shortest_path_distance(lcs):
      pprint(s2.tree(lambda s:s.hypernyms()+s.instance_hypernyms()))

    print()

  #return nltk_dist
  return custom_dist
  
def wup_similarity(s1: Synset, s2: Synset):
  lcs = find_LCS(s1, s2)
  if not lcs:
    return 0

  lcs_depth = lcs.min_depth() + 1
  depth1 = synsets_distance(s1, lcs) + lcs_depth
  depth2 = synsets_distance(s2, lcs) + lcs_depth
  return (2*lcs_depth) / (depth1 + depth2)

def sp_similarity(s1: Synset, s2: Synset):
  return 2*WORDNET_MAX_DEPTH - (synsets_distance(s1, s2) or 2*WORDNET_MAX_DEPTH)

def lch_similarity(s1: Synset, s2: Synset):
  return -np.log((synsets_distance(s1, s2) or 1.)/(2*WORDNET_MAX_DEPTH))

def get_similarity(s1: Synset, s2: Synset, method='wup'):
  similarity_methods = {
    'wup': wup_similarity,
    'sp': sp_similarity,
    'lch': lch_similarity
  }

  return similarity_methods[method](s1, s2)

  s1._pos = wordnet.NOUN
  s2._pos = wordnet.NOUN

  if method == 'wup':
    return s1.wup_similarity(s2) or 0.
  elif method == 'sp':
    return s1.path_similarity(s2) or 0.
  else:
    return s1.lch_similarity(s2) or 0.

def cov(X, Y):
  X = np.array(X)
  Y = np.array(Y)
  X_mean = X.mean()
  Y_mean = Y.mean()
 
  return 1/len(X) * ((X-X_mean)*(Y-Y_mean)).sum() 

def pearson_corr(X, Y):
  return cov(X, Y) / (np.std(X)*np.std(Y))

def spearman_corr(X, Y, num_bins=-1):
  X = np.array(X)
  Y = np.array(Y)

  if num_bins == -1:
    set_X = np.array(list(set(X)))
    set_Y = np.array(list(set(Y)))
    rank_X = {val:rank for (val, rank) in zip(set_X, np.argsort(set_X))}
    rank_Y = {val:rank for (val, rank) in zip(set_Y, np.argsort(set_Y))}
    return pearson_corr([rank_X[val] for val in X], [rank_Y[val] for val in Y])
  
  bins = np.linspace(0, 1, num_bins)
  return pearson_corr(np.digitize(X, bins), np.digitize(Y, bins))

def get_max_similarity(synsets1, synsets2, method):
  return max(map(
    lambda s: get_similarity(s[0], s[1], method=method), 
    itertools.product(synsets1, synsets2)))

if __name__ == '__main__':
  print(wordnet.get_version())

  term1, term2, ground_similarity = load_csv('./WordSim353.csv')
  norm_similarity = ground_similarity / 10.

  similarities = {
    'wup': [],
    'sp': [],
    'lch': []
  }
  
  for t1, t2, gs in zip(term1, term2, ground_similarity):
    print(f'{t1}-{t2}: human {gs:.3f} ', end='', flush=True)
    synsets1 = wordnet.synsets(t1)
    synsets2 = wordnet.synsets(t2)

    if len(synsets1) == 0 or len(synsets2) == 0:
      similarities['wup'].append(0)
      similarities['sp'].append(0)
      similarities['lch'].append(0)
      continue

    wup_sim = get_max_similarity(synsets1, synsets2, method='wup')
    sp_sim = get_max_similarity(synsets1, synsets2, method='sp')
    lch_sim = get_max_similarity(synsets1, synsets2, method='lch')
    similarities['wup'].append(wup_sim)
    similarities['sp'].append(sp_sim)
    similarities['lch'].append(lch_sim)

    print(f'wup: {wup_sim:.3f}, sp: {sp_sim:.3f}, lch: {lch_sim:.3f}')

  print('\n---------- WUP ----------')
  print('Pearson correlation: {:.3f}'.format(pearson_corr(similarities['wup'], norm_similarity)))
  print('Spearman correlation: {:.3f}'.format(spearman_corr(similarities['wup'], norm_similarity)))

  print('---------- SP -----------')
  print('Pearson correlation: {:.3f}'.format(pearson_corr(similarities['sp'], norm_similarity)))
  print('Spearman correlation: {:.3f}'.format(spearman_corr(similarities['sp'], norm_similarity)))

  print('---------- LCH -----------')
  print('Pearson correlation: {:.3f}'.format(pearson_corr(similarities['lch'], norm_similarity)))
  print('Spearman correlation: {:.3f}'.format(spearman_corr(similarities['lch'], norm_similarity)))

  print(f'Diverging distances: {wrong_distances}')
  print(f'Different LCSs: {diff_lcs}')


  a = [1,2,3,4,5,6]
  b = a[::-1] #[1,2,3,4,5,6]

  print(f"pearson({a}, {b}): {pearson_corr(a,b)}")
  print(f"spearman({a}, {b}): {spearman_corr(a,b)}")