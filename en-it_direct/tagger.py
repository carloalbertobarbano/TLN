import logging
import numpy as np
from treebank import TreeBank

logger = logging.getLogger(__name__)

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