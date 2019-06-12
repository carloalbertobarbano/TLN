import coloredlogs, logging

from treebank import TreeBank
from tagger import BaselinePOSTagger, MarkovPOSTagger

from translator import Translator

logger = logging.getLogger(__name__)

coloredlogs.install(level='INFO')

if __name__ == '__main__':
  train_set = TreeBank('./UD_English-ParTUT/en_partut-ud-train.conllu')
  test_set = TreeBank('./UD_English-ParTUT/en_partut-ud-test.conllu')
  dev_set = TreeBank('./UD_English-ParTUT/en_partut-ud-dev.conllu')

  baseline_tagger = BaselinePOSTagger()
  baseline_tagger.train(train_set)

  markov_tagger = MarkovPOSTagger()
  markov_tagger.train(train_set)

  sentence, tags = test_set[5] #['Secretariat', 'is', 'expected', 'to', 'race', 'tomorrow']
  print(f'Predicting: {sentence}')
  print(f'Tags: {tags}')

  baseline_tags = baseline_tagger.predict(sentence)
  markov_tags = markov_tagger.predict(sentence)

  print('Baseline prediction:', baseline_tags)
  print('Markov prediction:', markov_tags)

  accuracy = baseline_tagger.evaluate(train_set)
  print(f'Baseline - Accuracy on train set: {accuracy:.2f}')
  accuracy = baseline_tagger.evaluate(test_set)
  print(f'Baseline - Accuracy on test set: {accuracy:.2f}')

  accuracy = markov_tagger.evaluate(train_set)
  print(f'HMM - Accuracy on train set: {accuracy:.2f}')
  accuracy = markov_tagger.evaluate(test_set)
  print(f'HMM - Accuracy on test set: {accuracy:.2f}')

  print('Total number of ADJ:', train_set.tags['ADJ']['count']+test_set.tags['ADJ']['count']+dev_set.tags['ADJ']['count'])
  
  translator = Translator('./dict.txt', rules='./rules.txt')

  print()
  with open('./sentences_to_translate.txt', 'r') as file:
    lines = file.readlines()
  
  for line in lines:
    print('\n----------')
    print('EN:', line.strip())
    
    translation = translator.translate(line, baseline_tagger)
    logger.debug(f'Translation (Base): {translation}')
    print('IT (Base):', ' '.join([x[0] for x in translation]))
    
    translation = translator.translate(line, markov_tagger)
    logger.debug(f'Translation (HMM): {translation}')
    print('IT (HMM):', ' '.join([x[0] for x in translation]))
  