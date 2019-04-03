import numpy as np 

def parse_grammar(file):
  with open(file, 'r') as f:
    lines = f.readlines()
  
  lines = filter(lambda line: not line.startswith('#') and line.strip(), lines)
  cfg = map(lambda rule: tuple(rule.split('->')), lines)
 
  heads, productions = zip(
    *list(map(lambda rule: (rule[0].strip(), rule[1].strip()), cfg))
  )

  return list(heads), list(productions)

def CKY(cfg, sentence):
  heads, productions = cfg
  words = sentence.split(' ')
  matrix = np.empty((len(words), len(words)))

  for j in range(0, len(words)):
    matrix[j-1, j] = [head for head,production in zip(heads, productions) if production == words[j]]

    for i in range(j-2, 0):
      pass

  print(matrix)

  return None

cfg = parse_grammar('paolofrancesca.cfg')
print(cfg)

CKY(cfg, 'Paolo ama Francesca dolcemente')