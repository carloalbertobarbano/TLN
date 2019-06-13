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
    self.treebank = None

    self.smoothing_transition_unknown = 0.1
    self.smoothing_emission_pnoun = 1.
    self.smoothing_emission_unknown = 0.1

  def train(self, dataloader: TreeBank):
    super().train(dataloader)
    self.treebank = dataloader
    
    self.smoothing_transition_unknown = 1. / len(dataloader.tags.keys())
    self.smoothing_emission_unknown = 1. / len(dataloader.tags.keys())

    for data in dataloader:
      sentence, tags = data

      logger.debug(f'Training on sentence: {sentence}')
      logger.debug(f'Tags: {tags}')

      tags = ['start'] + tags 
      for i in range(len(tags)-1):
        tag = tags[i]
        succ = tags[i+1]

        if tag not in self.p_transition:
          logger.debug(f'Adding tag {tag} to transition table')
          self.p_transition[tag] =  {}
        if succ not in self.p_transition[tag]:
          self.p_transition[tag][succ] = 0.
        self.p_transition[tag][succ] += 1.

        
    for prev_tag in self.p_transition:
      for tag in self.p_transition[prev_tag]:
        if prev_tag == 'start':
          self.p_transition[prev_tag][tag] /= len(dataloader)
        else:
          self.p_transition[prev_tag][tag] /= dataloader.tags[prev_tag]['count']
        #self.p_transition[prev_tag][tag] = np.log(self.p_transition[prev_tag][tag] + 1.)
    
    #pprint(self.p_transition)

  def get_transition_prob(self, prev_state, state):
    if state not in self.p_transition[prev_state]:
      return self.smoothing_transition_unknown
      return 0.001
    return self.p_transition[prev_state][state]

  def get_known_emission_prop(self, state, token):
    if state == 'DET':
      dets = ['the', 'a', 'this', 'these', 'those', 'his', 'their', 'its', 'any', 'us', 'that', 'no']
      if token.lower() in dets:
        return 1. / len(dets)
    elif state == 'PRON':
      prons = ['it', 'that', 'he', 'i', 'we', 'which', 'they', 'you', 'this', 'who']
      if token.lower() in prons:
        return 1. / len(prons)
    return -1.

  def get_emission_prob(self, state, token):
    known_prob = self.get_known_emission_prop(state, token)
    if known_prob != -1:
      return known_prob

    if token not in self.treebank.tags[state]['emission']:
      if token[0].isupper() and state == 'PROPN':
        return self.smoothing_emission_pnoun
      return self.smoothing_emission_unknown 
    return self.treebank.tags[state]['emission'][token]

  def tune_hyperparams(self, dataloader, vmax=1., vmin=1e-10):
    best_accuracy = 0.
    best_hyp1 = best_hyp2  = best_hyp3 = vmax
    hyp1 = hyp2 = hyp3 = vmax

    logger.info(f'Performing grid search for hyperparams [{vmax} - {vmin}]')
    while hyp1 > vmin:
      hyp2 = vmax
      while hyp2 > vmin:
        hyp3 = vmax
        while hyp3 > vmin:
          self.smoothing_emission_pnoun = hyp1
          self.smoothing_emission_unknown = hyp2
          self.smoothing_transition_unknown = hyp3
          accuracy = self.evaluate(dataloader)
          if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_hyp1, best_hyp2, best_hyp3 = hyp1, hyp2, hyp3
          hyp3 /= 10.
        hyp2 /= 10.
      hyp1 /= 10.

    logger.info(f'Best hyperparams values: hyp1:{best_hyp1} hyp2:{best_hyp2} hyp3: {best_hyp3}')
    self.smoothing_emission_pnoun = best_hyp1
    self.smoothing_emission_unknown = best_hyp2
    self.smoothing_transition_unknown = best_hyp3

  def predict(self, sentence):
    viterbi = [{}]
    states = sorted(list(self.p_transition.keys()))

    for state in states:
      if state == 'start':
        continue
      p_transition = self.get_transition_prob('start', state)
      p_emission = self.get_emission_prob(state, sentence[0])
      viterbi[0][state] = {'prob': p_transition * p_emission,
                           'back': 'start'} 


    for i in range(1, len(sentence)):
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

    backtrace = max(viterbi[-1].items(), key=lambda x: x[1]['prob'])[0]

    path = []
    for i in range(len(sentence)-1, 0, -1):
      path.append(backtrace)
      backtrace = viterbi[i][backtrace]['back']
    path.append(backtrace)

    return path[::-1]










  