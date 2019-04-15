def loadNasari(path):
  with open(path, 'r') as file:
    lines = file.readlines()

  lines = filter(lambda line: not line.startswith('#') and line.strip(), lines)
  nasari = {}

  for line in lines:
    vec = line.split(';')
    babel_id = vec[0]
    word = vec[1]
    synsets = filter(lambda s: s and s.strip(), vec[2:])

    nasari[word] = {'babel_id': babel_id, 'synsets': {}}
    for synset in synsets:
      if '_' not in synset:
        continue
      #print(synset)
      synset_name, synset_weight = synset.split('_')
      nasari[word]['synsets'][synset_name] = float(synset_weight)

  return nasari


if __name__ == '__main__':
  nasari = loadNasari('./dd-small-nasari-15.txt')
  print(nasari['Pussy'])