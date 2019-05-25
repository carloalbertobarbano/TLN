from typing import List
import argparse

def load_nasari(path):
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

def load_text(path):
  with open(path, 'r') as file:
    lines = file.readlines()
  lines = list(filter(lambda line: not line.startswith('#') and line.strip(), lines))
  return lines

def summarize_text(paragraphs: List[str], compression=10):
  return paragraphs

parser = argparse.ArgumentParser()
parser.add_argument('-i', help='input file path', type=str)
parser.add_argument('-o', help='output path', type=str)
parser.add_argument('-c', help='compression rate', type=int, default=10)

if __name__ == '__main__':
  args = parser.parse_args()

  nasari = load_nasari('./dd-small-nasari-15.txt')
  text = load_text(args.i)
  summarized_text = summarize_text(text, args.c)

  with open(args.o, 'w') as file:
    for paragraph in summarized_text:
      file.write(paragraph)
      file.write('\n')