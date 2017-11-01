import gensim, logging
import re
import os

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

special_character=re.compile('[^a-zA-Z0-9\n\.]')
url_regex= re.compile(r'^https?:\/\/.*[\r\n]*',flags=re.MULTILINE)
numbers_reg=re.compile(r'\d+')

class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname
 
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname),encoding = "ISO-8859-1"):
                try:
                    #line = gensim.utils.to_unicode( line, encoding='utf-8', errors='replace')
                    line = special_character.sub(' ',line)
                    line = url_regex.sub(' ',line)
                    line = numbers_reg.sub('n',line)
                    line = line.lower()
                    yield line.split()
                except:
                    print('error',line)
                    break
            
                

#sentences = MySentences('/Users/jayavardhanreddy/Github_Data_Files/Word2vec_Files/') # a memory-friendly iterator
sentences = MySentences('/Users/jayavardhanreddy/Github_Data_Files/w2v/w2v_ver2/')
model = gensim.models.Word2Vec(sentences,workers=4)
model.save('/Users/jayavardhanreddy/Github_Data_Files/created_w2v_ver2.model')

