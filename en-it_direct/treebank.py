import coloredlogs, logging
import numpy as np

logger = logging.getLogger(__name__)

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
      if tag == '_':
        continue
      
      if tag not in self.tags:
        self.tags[tag] = {'count': 0., 
                          'emission': {}} 

      if word not in self.tags[tag]['emission']:
        self.tags[tag]['emission'][word] = 0.

      self.tags[tag]['count'] += 1.
      self.tags[tag]['emission'][word] += 1.
      self.sentences[-1]['tags'].append(tag)
      self.sentences[-1]['tokens'].append(word)
    
    for tag in self.tags:
      for word in self.tags[tag]['emission']:
        self.tags[tag]['emission'][word] /= self.tags[tag]['count']
        #self.tags[tag]['emission'][word] = np.log(self.tags[tag]['emission'][word] + 1.)

  def __len__(self):
    return len(self.sentences)

  def __getitem__(self, index):
    return self.sentences[index]['tokens'], self.sentences[index]['tags']
