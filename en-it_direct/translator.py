import nltk
import logging

from tagger import BaselinePOSTagger

logger = logging.getLogger(__name__)

class Translator:
  def __init__(self, dictionary, tagger: BaselinePOSTagger, rules=None):
    with open(dictionary, 'r') as file:
      lines = file.readlines()
    self.dictionary = {}

    for line in lines:
      tok = line.strip().split('\t')
      self.dictionary[tok[0]] = tok[1]
    
    self.tagger = tagger
    self.rules = []
    if rules is not None:
      with open(rules, 'r') as file:
        lines = file.readlines()
      
      for line in lines:
        left, right = line.strip().split('\t')
        left = left.split(' ')
        right = right.split(' ')
        self.rules.append((left, [int(x) for x in right]))
        logger.info(f'Read rule: {self.rules[-1]}')

  def tokenize_sentence(self, sentence):
    return nltk.word_tokenize(sentence)

  def check_if_rule_applies(self, rule, tags):
    for i in range(len(tags) - len(rule) + 1):
      if rule == tags[i:i+len(rule)]: return i
    return -1

  def translate(self, sentence):
    tokens = self.tokenize_sentence(sentence)
    tags = self.tagger.predict(tokens)

    logger.info(f'Tagged sentence: {list(zip(tokens, tags))}')

    translation = []
    for token in tokens:
      if token.lower() in self.dictionary:
        token = self.dictionary[token.lower()]
      translation.append(token)
    
    #rules = [(['NOUN', 'PART-\'s', 'NOUN'], [2, 1, 0])]
    for rule in self.rules:
      rule_tags = rule[0]
      rule_tags = list(map(lambda tag: tag.split('-')[0], rule_tags))
      swap_indices = rule[1]

      index = self.check_if_rule_applies(rule_tags, tags)
      if index >= 0:
        logger.info(f'Found matching rule: {rule}')
        tmp = translation.copy()
        tmp_tags = tags.copy()

        for i, swap in enumerate(swap_indices):
          logger.info(f'swapping {tmp[index]} with {translation[index+swap]}')
          tmp[index+i] = translation[index+swap]
          tmp_tags[index+i] = tags[index+swap]
        
        translation = tmp.copy()
        tags = tmp_tags.copy()
        logger.info(f'New sentence: {list(zip(translation, tags))}')

    return list(zip(translation, tags))
      
