def loadNasari(path):
  with open(path, 'r') as file:
    lines = file.readlines()

  lines = filter(lambda line: not line.startswith('#') and line.strip(), lines)
  lines = [line.split(';') for line in lines]
  nasari = {}

  for vec in lines:
    babel_id = vec[0]
    word = vec[1]
    synsets = filter(lambda s: s and s.strip(), vec[2:])
    nasari[word] = {'babel_id': babel_id, 'synsets': {}}

    synsets = list(filter(lambda synset: '_' in synset, synsets))
    for synset in synsets:
      synset_name, synset_weight = synset.split('_')
      nasari[word]['synsets'][synset_name] = float(synset_weight)

  return nasari


if __name__ == '__main__':
  nasari = loadNasari('./dd-small-nasari-15.txt')
  print(nasari['Pussy'])