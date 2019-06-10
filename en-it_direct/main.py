import coloredlogs, logging

from treebank import TreeBank
from tagger import BaselinePOSTagger, MarkovPOSTagger

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG')

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
  