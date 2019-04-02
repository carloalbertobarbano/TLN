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
  return None

cfg = parse_grammar('paolofrancesca.cfg')
print(cfg)