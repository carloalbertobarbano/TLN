from itertools import starmap
from nltk.corpus import wordnet

def load_csv(path_csv):
  with open(path_csv, 'r') as file:
    lines = file.readlines()
  lines = lines[1:]

  term1, term2, similarity = zip(*map(lambda l: l.split(','), lines))
  return term1, term2, similarity

def find_LCS(s1, s2):
  pass

def get_similarity(s1, s2, method='wup'):
  return s1.wup_similarity(s2)

if __name__ == '__main__':
  term1, term2, similarity = load_csv('./WordSim353.csv')
  print(term1[0], ' ', term2[0], ' ', similarity[0])

  syns = wordnet.synsets(term2[0])
  print(syns[0].name())
  similarity = []

  for t1, t2, in zip(term1, term2):
    synsets1 = wordnet.synsets(t1)
    synsets2 = wordnet.synsets(t2)

    if len(synsets1) == 0 or len(synsets2) == 0:
      similarity.append(0)
      continue
    
    sim = list(map(lambda s: get_similarity(s[0], s[1]), zip(synsets1, synsets2)))
    print(t1, "-", t2)
    similarity.append(max(sim))

  print(similarity)

  w1 = wordnet.synsets(term1[0])[0]
  w2 = wordnet.synsets(term2[0])[0]
  print(w1.wup_similarity(w2))
  print(w1.min_depth())
  print(w1.max_depth())

  for hypernym in w2.hypernyms():
    print(hypernym)
