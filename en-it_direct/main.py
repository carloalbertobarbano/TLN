import coloredlogs, logging
import numpy as np

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG')

class TreeBank:
  def __init__(self, path):
    self.path = path
    self.tags = {}
    self.sentences = []

    self.parse_treebank(path)
    logger.info(f'Loaded {len(self.sentences)} sentences and {len(self.tags.keys())} tags')
    logger.info(f'Tags: {sorted(self.tags.keys())}')
    
  def parse_treebank(self, path):
    with open(path, 'r') as file:
      lines = file.readlines()
    
    for line in lines:
      if line.startswith('#'):
        if "sent_id" in line:
          self.sentences.append({'sent_id': line.split('=')[1].strip(), 
                                 'tags': []})  
        elif "text =" in line:
          self.sentences[-1]['text'] = line.split('=')[1].strip()
          self.sentences[-1]['tokens'] = []
        continue
        
      if not line.strip():
        continue
      
      data = line.strip().split('\t')
      word, tag = data[1], data[3] #word tolower?
      
      if tag not in self.tags:
        self.tags[tag] = {'count': 0, 
                          'emission': {}} 

      if word not in self.tags[tag]['emission']:
        self.tags[tag]['emission'][word] = 0

      self.tags[tag]['count'] += 1
      self.tags[tag]['emission'][word] += 1
      self.sentences[-1]['tags'].append(tag)
      self.sentences[-1]['tokens'].append(word)

  def __len__(self):
    return len(self.sentences)

  def __getitem__(self, index):
    return self.sentences[index]['tokens'], self.sentences[index]['tags']

class BaselinePOSTagger:
  def __init__(self):
    self.words = {}
  
  def train(self, dataloader: TreeBank):
    # Initializes tags frequencies for words
    for data in dataloader:
      sentence, tags = data

      for token, tag in zip(sentence, tags):
        if token not in self.words:
          self.words[token] = {}

        if tag not in self.words[token]:
          self.words[token][tag] = 0
        self.words[token][tag] += 1
  
  def is_punctuation(self, token):
    punctuation = ".,;:!?"
    for p in punctuation:
      if token == p:
        return True
    return False

  def predict(self, sentence):
    # Get most frequent tag for given word, NOUN if unknown
    tags = []
    for token in sentence:
      tag = 'NOUN'

      if token.count('.') <= 1 and token.replace('.', '').isdigit():
        tag = 'NUM'
      elif self.is_punctuation(token):
        tag = 'PUNCT'
      elif token in self.words:
        tag = max(self.words[token].items(), key=lambda x: x[1])[0]
      tags.append(tag)
    return tags

  def evaluate(self, dataloader):
    accuracy = 0.
    num_tags = 0.

    for data in dataloader:
      sentence, tags = data
      prediction = self.predict(sentence)

      tags = np.array(tags)
      prediction = np.array(prediction)

      accuracy += np.count_nonzero(tags == prediction)
      num_tags += len(tags)
    
    return accuracy / num_tags

class MarkovPOSTagger(BaselinePOSTagger):
  pass

if __name__ == '__main__':
  train_set = TreeBank('./UD_English-ParTUT/en_partut-ud-train.conllu')
  test_set = TreeBank('./UD_English-ParTUT/en_partut-ud-test.conllu')

  baseline_tagger = BaselinePOSTagger()
  baseline_tagger.train(train_set)
  tags = baseline_tagger.predict(test_set[5][0])

  accuracy = baseline_tagger.evaluate(train_set)
  print(f'Accuracy on train set: {accuracy:.2f}')
  accuracy = baseline_tagger.evaluate(test_set)
  print(f'Accuracy on test set: {accuracy:.2f}')
  