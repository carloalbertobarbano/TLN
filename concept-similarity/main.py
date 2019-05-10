import itertools
import numpy as np
from nltk.corpus import wordnet

def load_csv(path_csv):
  with open(path_csv, 'r') as file:
    lines = file.readlines()
  lines = lines[1:]

  term1, term2, similarity = zip(*map(lambda l: l.split(','), lines))
  return term1, term2, np.array([float(s.strip()) for s in similarity])

def find_LCS(s1, s2):
  pass

def get_similarity(s1, s2, method='wup'):
  sim = s1.wup_similarity(s2)
  if sim is None:
    sim = 0
  return sim

def cov(X, Y):
  X = np.array(X)
  Y = np.array(Y)
  X_mean = X.mean()
  Y_mean = Y.mean()
 
  return 1/len(X) * ((X-X_mean)*(Y-Y_mean)).sum() 

def pearson_corr(X, Y):
  return cov(X, Y) / (np.std(X)*np.std(Y))

def spearman_corr(X, Y):
  X = np.array(X)
  Y = np.array(Y)
  return pearson_corr(np.argsort(X), np.argsort(Y))
  #bins = np.array([0., 3., 6., 9.])
  #return pearson_corr(np.digitize(X*10, bins), np.digitize(Y*10, bins))
  #return pearson_corr((np.array(X)*10).astype(int), (np.array(Y)*10).astype(int))

if __name__ == '__main__':
  term1, term2, ground_similarity = load_csv('./WordSim353.csv')
  norm_similarity = ground_similarity / 10.

  similarity = []
  for t1, t2, gs in zip(term1, term2, ground_similarity):
    print(f'{t1}-{t2}: human {gs:.3f} ', end='', flush=True)
    synsets1 = wordnet.synsets(t1)
    synsets2 = wordnet.synsets(t2)

    if len(synsets1) == 0 or len(synsets2) == 0:
      similarity.append(0)
      continue
    
    wup_sim = list(map(lambda s: get_similarity(s[0], s[1]), itertools.product(synsets1, synsets2)))
    wup_sim = max(wup_sim)
    similarity.append(wup_sim)

    print(f'wup: {wup_sim:.3f}')

  print('----------')
  print('Pearson correlation: {:.3f}'.format(pearson_corr(similarity, norm_similarity)))
  print('Spearman correlation: {:.3f}'.format(spearman_corr(similarity, norm_similarity)))

  #bins = np.array([0., 2.5, 5., 7.5, 10.])
  #print(np.digitize(np.array(similarity)*10, bins))
  #print((np.array(similarity)*10).astype(int))
  #print(np.array(ground_similarity).astype(int))
  #for hypernym in w2.hypernyms():
  #  print(hypernym)
