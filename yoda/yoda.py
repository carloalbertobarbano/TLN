import numpy as np 
from pprint import pprint

def parse_grammar(file):
  with open(file, 'r') as f:
    lines = f.readlines()
  
  lines = filter(lambda line: not line.startswith('#') and line.strip(), lines)
  cfg = map(lambda rule: tuple(rule.split('->')), lines)
 
  heads, productions = zip(
    *list(map(lambda rule: (rule[0].strip(), rule[1].strip()), cfg))
  )

  return list(heads), list(productions)

def find_heads(cfg, production):
  print(f"Searching cfg for production {production}")
  
  rules = list(filter(lambda rule: rule[1] == production, zip(*cfg)))
  if not rules:
    return list()

  heads, production = zip(*rules)
  return list(heads)

def CKY(cfg, sentence):
  heads, productions = cfg
  words = sentence.split(' ')
  N = len(words)
  matrix = np.empty((N, N), dtype=object)
  for i in range(N):
    for j in range(N):
      matrix[i, j] = list()
  #matrix = [[None]*N]*N
  pprint(matrix)

  for j in range(0, N):
    tags = find_heads(cfg, words[j])
    print(f"Tags for {words[j-1]}: {tags}")
    matrix[j][j] = tags

    pprint(matrix)
    for i in range(j-1, -1, -1):
      for k in range(i+1, j+1):
        print(f"i: {i} k: {k}")
        B = matrix[i, k]
        C = matrix[k, j-1]

        for b in B:
          for c in C:
            print(f"i:{i}, k:{k}, j:{j} b: {b}, c: {c}")
            A = find_heads(cfg, "{} {}".format(b, c))
            if matrix[i, j-1] is None:
              matrix[i, j-1] = list()
            matrix[i, j-1].append(A)

  return matrix

cfg = parse_grammar('paolofrancesca.cfg')
print(cfg)

matrix = CKY(cfg, 'Paolo ama Francesca dolcemente')
pprint(matrix)