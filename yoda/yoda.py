import numpy as np 
import copy
from pprint import pprint

import sys

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
  tag = None

  def __init__(self, symbol, children):
    self.symbol = symbol
    self.children = children
    self.tag = None #''

  def __repr__(self):
    s = self.symbol + ('[' + self.tag + ']' if self.tag else '')
    #return s

    if self.children:
      s += '('
      for child in self.children:
        s += ' ' + repr(child)
      s += ')'
    return s

  def __str__(self):
    return self.symbol

  def print_sentence(self):
    if not self.children:
      print(self.symbol + ' ', end='')
    else:
      for child in self.children:
        child.print_sentence()

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
        symbol=tag, children=[Constituent(symbol=words[j], children=[])]
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

# Rules: 
# ['S -> S VX',
#  'VP -> V X']
def tag_svx(S, rules):
  for rule in rules:
    head, tags = map(lambda s: s.strip(), rule.split('->'))
    tags = tags.split(' ')

    if S.symbol == head:
      S.children[0].tag = tags[0]
      S.children[1].tag = tags[1]
    
  for child in S.children:
    tag_svx(child, rules)

def svx_sxv(S):
  if not S.children or len(S.children) < 2:
    return

  svx_sxv(S.children[0])
  svx_sxv(S.children[1])

  if S.children[0].tag == 'V' and S.children[1].tag == 'X':
    S.tag = 'XV'
    S.children.reverse()

    if S.children[0].children and len(S.children[0].children) > 1:
      if S.children[0].children[0].symbol == 'ADV' and S.children[0].children[1].symbol == 'NP':
        S.children[0].children.reverse()

def sxv_xsv(S):
  if not S.children or len(S.children) < 2:
    return
  
  sxv_xsv(S.children[0])
  sxv_xsv(S.children[1])

  if S.tag == 'XV':
    S.tag = 'SV'

  if S.children[0].tag == 'S' and S.children[1].children[0].tag == 'X':
    s = S.children[0]
    x = S.children[1].children[0]
    S.children[0], S.children[1].children[0] = x, s
  
def transfer(src: Constituent):
  res = copy.deepcopy(src)
  
  # Step 1. SVX -> SXV
  svx_sxv(res)
  #print('svx->sxv: ')
  #res.print_sentence()
  
  # Step 2. SXV -> XSV
  sxv_xsv(res)
  #print('\nsxv->xsv: ')
  #res.print_sentence()
  
  return res

if __name__ == '__main__':
  cfg = parse_grammar('G2.cfg')
  print(cfg)
  print()

  with open('./sentences.txt', 'r') as file:
    sentences = file.readlines()
  
  for sentence in sentences:
    sentence = sentence.strip()
    matrix = CKY(cfg, sentence)
    
    S = matrix[0, -1][0]
    S.print_sentence()
    print()

    tag_svx(S, ['S -> S VX', 'VP -> V X'])
    print(repr(S))

    res = transfer(S)
    print(repr(res))
    res.print_sentence()
    print("\n")