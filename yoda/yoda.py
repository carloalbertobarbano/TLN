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

class Node:
  father = None
  children = list()
  symbol = None

  def __init__(self, symbol, father, children):
    print(f"node {symbol}")
    self.symbol = symbol
    self.father = father
    self.children = children

  def __repr__(self):
    s = self.symbol
    return s
    if self.children:
      s += '('
      for child in self.children:
        s += '(' + repr(child) + ') '
      s += ')'
    return s

  def __str__(self):
    return self.symbol

def find_heads(cfg, production):
  print(f"Searching cfg for production {production}")
  
  rules = list(filter(lambda rule: rule[1] == production, zip(*cfg)))
  if not rules:
    return list()

  print(f"Found rules {rules}")
  heads, production = zip(*rules)
  return list(heads)

def CKY(cfg, sentence):
  words = sentence.split(' ')
  N = len(words)
  matrix = np.empty((N, N), dtype=object)
  for i in range(N):
    for j in range(N):
      matrix[i, j] = list()
  pprint(matrix)

  for j in range(0, N):
    tags = find_heads(cfg, words[j])
    print(f"Tags for {words[j-1]}: {tags}")
    matrix[j][j] = list(map(lambda tag: Node(symbol=tag, children=[], father=None), tags))

    print(matrix)
    for i in range(j-1, -1, -1):
      for k in range(i+1, j+1):
        B = matrix[i, k-1]
        C = matrix[k, j]

        print(f"B[{i}, {k}]={B}, C[{k}, {j}]={C}")
        heads = list()

        for b in B:
          for c in C:
            A = find_heads(cfg, "{} {}".format(b, c))
            heads.extend(list(map(lambda head: Node(symbol=head, children=[B, C], father=None), A)))
        
        matrix[i, j].extend(heads)

  return matrix

cfg = parse_grammar('paolofrancesca.cfg')
print(cfg)

matrix = CKY(cfg, 'Paolo ama Francesca dolcemente')
pprint(matrix)
pprint(matrix[0, -1][0].children)