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

class Constituent:
  children = None
  symbol = None

  def __init__(self, symbol, children):
    self.symbol = symbol
    self.children = children

  def __repr__(self):
    s = self.symbol

    if self.children:
      s += '('
      for child in self.children:
        s += ' ' + repr(child)
      s += ')'
    return s

  def __str__(self):
    return self.symbol

def find_heads(cfg, production):  
  rules = list(filter(lambda rule: rule[1] == production, zip(*cfg)))
  if not rules:
    return list()

  heads, production = zip(*rules)
  return list(heads)

def CKY(cfg, sentence):
  words = sentence.split(' ')
  N = len(words)
  matrix = np.empty((N, N), dtype=object)
  for i in range(N):
    for j in range(N):
      matrix[i, j] = list()

  for j in range(0, N):
    tags = find_heads(cfg, words[j])
    matrix[j][j] = list(map(
      lambda tag: Constituent(
        symbol=tag, children=[]
      ), tags
    ))

    for i in range(j-1, -1, -1):
      for k in range(i+1, j+1):
        B = matrix[i, k-1]
        C = matrix[k, j]
        heads = list()

        for b in B:
          for c in C:
            A = find_heads(cfg, f"{b} {c}")
            heads.extend(list(map(lambda head: Constituent(symbol=head, children=[b, c]), A)))
        
        matrix[i, j].extend(heads)

  return matrix

cfg = parse_grammar('paolofrancesca.cfg')
print(cfg)

matrix = CKY(cfg, 'Paolo ama Francesca dolcemente')
pprint(matrix[0, -1][0])

yoda_cfg = parse_grammar('yoda.cfg')
