import requests
import json

with open('wordlist_babelnet.txt', 'r') as file:
  lines = file.readlines()

words = {}
for line in lines:
  word = line.strip()
  r = requests.get('https://babelnet.io/v5/getSynsetIds?lemma={}&searchLang=EN&key=558130e3-cf9a-4501-86ce-89ae96041923'.format(word))
  response = r.json()
  print(response)

  words[word] = []
  for id in response:
    words[word].append(id['id'])

with open('babelnet_ids.json', 'w') as file:
  json.dump(words, file)


