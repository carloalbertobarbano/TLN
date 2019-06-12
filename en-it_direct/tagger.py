import logging
import numpy as np

from pprint import pprint
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

  def evaluate(self, dataloader: TreeBank):
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
  def __init__(self):
    super().__init__()
    self.p_transition = {}
    self.p_emission = {}

  def train(self, dataloader: TreeBank):
    super().train(dataloader)

    for data in dataloader:
      sentence, tags = data

      logger.debug(f'Training on sentence: {sentence}')
      logger.debug(f'Tags: {tags}')

      for token, tag in zip(sentence, tags):
        if tag not in self.p_emission:
          self.p_emission[tag] = {}
        if token not in self.p_emission[tag]:
          self.p_emission[tag][token] = 0.
        self.p_emission[tag][token] += 1.

      tags = ['start'] + tags 
      for i in range(len(tags)-1):
        tag = tags[i]
        succ = tags[i+1]

        if tag not in self.p_transition:
          print(f'Adding tag {tag} to transition table')
          self.p_transition[tag] =  {}
        if succ not in self.p_transition[tag]:
          self.p_transition[tag][succ] = 0.
        self.p_transition[tag][succ] += 1.

        
    for prev_tag in self.p_transition.keys():
      for tag in self.p_transition[prev_tag].keys():
        self.p_transition[prev_tag][tag] /= dataloader.tags[tag]['count']
        #self.p_transition[prev_tag][tag] = np.log(self.p_transition[prev_tag][tag])

    for tag in self.p_emission.keys():
      for token in self.p_emission[tag].keys():
        continue
        self.p_emission[tag][token] /= dataloader.tags[tag]['count']
        #self.p_emission[tag][token] = np.log(self.p_emission[tag][token])
    
    #pprint(self.p_transition)
    #pprint(self.p_emission)

  def get_transition_prob(self, prev_state, state):
    if state not in self.p_transition[prev_state]:
      return 1. / len(self.p_transition.keys())
    return self.p_transition[prev_state][state]

  def get_known_emission_prop(self, state, token):
    if state == 'DET':
      if token.lower() in ['the', 'a', 'this', 'his', 'their', 'its', 'any', 'us', 'that', 'no']:
        return 1.
    elif state == 'PRON':
      if token.lower() in ['it', 'that', 'he', 'i', 'we', 'which', 'they', 'you', 'this', 'who']:
        return 1.
    return -1.

  def get_emission_prob(self, state, token):
    known_prob = self.get_known_emission_prop(state, token)
    #if known_prob == 1.:
    #  return 1.

    if token not in self.p_emission[state]:
      #if token[0].isupper() and state == 'PROPN':
      #  return 1.
      return 1. / len(self.p_transition.keys())
    return self.p_emission[state][token]

  def predict(self, sentence):
    #print('PREDICTING SENTENCE: ', sentence)
    viterbi = [{}]
    states = sorted(list(self.p_transition.keys()))
    #print('States: ', states)

    for state in states:
      if state == 'start':
        continue
      p_transition = self.get_transition_prob('start', state)
      p_emission = self.get_emission_prob(state, sentence[0])
      viterbi[0][state] = {'prob': p_transition*p_emission,
                           'back': 'start'} 

    for i in range(1, len(sentence)):
      #print(f'Token: {sentence[i]}')
      viterbi.append({})
      for state in states:
        if state == 'start':
          continue
        p_emission = self.get_emission_prob(state, sentence[i])
        max_prob, max_state = -np.inf, None
        
        for prev_state in states:
          if prev_state == 'start':
            continue
          p_transition = self.get_transition_prob(prev_state, state)
          prob = p_transition * viterbi[i-1][prev_state]['prob']

          if prob > max_prob:
            max_prob = prob
            max_state = prev_state
        
        max_prob = max_prob * p_emission
        viterbi[i][state] = {'prob': max_prob, 'back': max_state}
    
    #print()
    #print('Last column of viterbi:')
    #pprint(viterbi[-1])

    #for i in range(0, len(sentence)):
    #  print(f'Column {i} of viterbi (token: {sentence[i]}')
    #  pprint(viterbi[i])

    backtrace = max(viterbi[-1].items(), key=lambda x: x[1]['prob'])[0]
    #print('END OF SENTENCE: ', backtrace)

    path = []
    for i in range(len(sentence)-1, 0, -1):
      path.append(backtrace)
      backtrace = viterbi[i][backtrace]['back']
    path.append(backtrace)

    #print('TAGS:', path[::-1])
    #print(f'Len TAGS: {len(path)}, len tokens: {len(sentence)}')
    return path[::-1]










  