import nltk
import logging
import re

from tagger import BaselinePOSTagger

logger = logging.getLogger(__name__)

class Translator:
  def __init__(self, dictionary, rules=None):
    with open(dictionary, 'r') as file: #encoding = "ISO-8859-1"
      lines = file.readlines()
    self.dictionary = {}

    for line in lines:
      if line.startswith('#'):
        continue
      tok = line.strip().split('\t')
      self.dictionary[tok[0]] = tok[1]
    
    self.rules = []
    if rules is not None:
      with open(rules, 'r') as file:
        lines = file.readlines()
      
      for line in lines:
        if not line.strip():
          continue

        left, right = line.strip().split('\t')
        left = left.split(' ')
        right = right.split(' ')
        self.rules.append((left, right))
        logger.info(f'Read rule: {self.rules[-1]}')

  def tokenize_sentence(self, sentence):
    return nltk.word_tokenize(sentence)

  def check_if_rule_applies(self, rule, tags, tokens, rule_literals):
    if len(rule) != len(rule_literals):
      logger.error(f'Length of rule {len(rule)} and of rule literals {len(rule_literals)} differ')

    rule = list(map(lambda r: r.replace('.', ''), rule))
    for i in range(len(tags) - len(rule) + 1):
      if rule == tags[i:i+len(rule)]: 
        token_match = list(filter(lambda pair: re.match(pair[1], pair[0]),
                                  zip(tokens[i:i+len(rule)], rule_literals)))
        if len(token_match) == len(rule):
          return i
    return -1

  def translate(self, sentence, tagger: BaselinePOSTagger):
    tokens = self.tokenize_sentence(sentence)
    tags = tagger.predict(tokens)

    print(' '.join(["{}[{}]".format(x[0],x[1]) for x in list(zip(tokens, tags))]))

    #rules = [(['NOUN', 'PART-\'s', 'NOUN'], [2, 1, 0])]
    for rule in self.rules:
      logger.debug(f'Checking rule {rule}')
      rule_body = rule[0]
      rule_tags = list(map(lambda tag: tag.split('-')[0], rule_body))
      rule_literals = list(map(lambda tag: tag.split('-')[1] if '-' in tag else '.*', rule_body))
      
      swap_indices = list(map(lambda index: index.split('-')[0], rule[1]))
      swap_literals = list(map(lambda index: index.split('-')[1] if '-' in index else None, rule[1]))

      index = self.check_if_rule_applies(rule_tags, tags, tokens, rule_literals)
      
      tmp_rule_tags, tmp_rule_literals = [],[]
      tmp_swap_indices, tmp_swap_literals = [],[]
      if index == -1:
        for i in range(len(rule_tags)):
          if '.' not in rule_tags[i]:
            tmp_rule_tags.append(rule_tags[i])
            tmp_rule_literals.append(rule_literals[i])
          
          if '.' not in swap_indices[i]:
            tmp_swap_indices.append(swap_indices[i])
            tmp_swap_literals.append(swap_literals[i])
        rule_tags, rule_literals = tmp_rule_tags, tmp_rule_literals
        swap_indices, swap_literals = tmp_swap_indices, tmp_swap_literals

        index = self.check_if_rule_applies(rule_tags, tags, tokens, rule_literals)

      if index >= 0:
        logger.info(f'Found matching rule: {rule}')
        tmp = tokens.copy()
        tmp_tags = tags.copy()

        for i, swap in enumerate(swap_indices):
          if swap == '+':
            logger.debug(f'Adding literal {swap_literals[i]}')
            new_token = re.findall(r"^\w+", swap_literals[i])[0]
            new_tag = re.findall(r"\[(.*?)\]", swap_literals[i])[0]
            
            tmp.insert(index+i, new_token)
            tmp_tags.insert(index+i, new_tag)
          else:
            swap = int(swap.replace('.', ''))
            logger.debug(f'swapping {tmp[index+i]} with {tokens[index+swap]}')
            tmp[index+i] = tokens[index+swap]
            tmp_tags[index+i] = tags[index+swap]

            if swap_literals[i] == '~':
              logger.debug(f'Removing {tokens[index+i]}[{tags[index+i]}]')
              del tmp[index+i]
              del tmp_tags[index+i]
              del swap_literals[i]
              del swap_indices[i]

            elif swap_literals[i]:
              logger.debug(f'Literal substitution {tokens[index+i]} with {swap_literals[i]}')
              tmp[index+i] = swap_literals[i]
        
        tokens = tmp.copy()
        tags = tmp_tags.copy()
        logger.debug(f'New sentence: {list(zip(tokens, tags))}')
    
    translation = []
    for token in tokens:
      if token.lower() in self.dictionary:
        token = self.dictionary[token.lower()]
      translation.append(token)


    return list(zip(translation, tags))
      
