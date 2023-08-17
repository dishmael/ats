import re
import magic
import pandas as pd
import PyPDF2

from collections import Counter
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from scipy import spatial

# This only needs to run once
#import nltk
#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('averaged_perceptron_tagger')

class Scanner():
  def __init__(self):
    pass

  def __str__(self):
    pass

  # Extract the text from the document
  def extract_text(self, fileName:str) -> str | None:
    type = magic.from_file(fileName, mime=True)
    text = ""

    match type:
      case 'application/pdf':
        with open(fileName, 'rb') as file:
          pdfReader = PyPDF2.PdfReader(file)
          for page in pdfReader.pages:
            text += page.extract_text()

      case 'text/plain':
        with open(fileName, 'r') as file:
          text += file.read()

      case _:
        print(f'Unknown File Type')
        return None

    return text

  # Extract the keywords from the text
  def extract_keywords(self, text:str) -> list[any]:
    stemmer = SnowballStemmer('english')
    wnl = WordNetLemmatizer()
    words = re.findall(r'\b\w+\b', text.lower())
    return [wnl.lemmatize(word) for word in words 
            if word not in stopwords.words('english') and len(word) > 2]
  
  # Score the resume against the job description (JD)
  def score(self, jdFile:str, resumeFile:str) -> float:
    # Evaluate the JD
    jdText = self.extract_text(jdFile)
    jdKeywords = self.extract_keywords(jdText)
    jdCounter = Counter(jdKeywords)
    jdCommon = {item[0]: item[1] for item in jdCounter.most_common(20)}

    # Evaluate the Resume
    resText = self.extract_text(resumeFile)
    resKeywords = self.extract_keywords(resText)
    resCounter = Counter(resKeywords)
    resCommon = [resCounter.get(key, 0) for key in jdCommon]

    data = {'JD': jdCounter, 'Resume': resCounter}
    df = pd.DataFrame(data)
    dfSorted = df.sort_values(by='JD', ascending=False)
    print(dfSorted)
    
    # Score based on similarity
    return 1 - spatial.distance.cosine(list(jdCommon.values()), resCommon)

def main():
  appName = 'Applicant Tracking System'
  
  sampleJobDescr = 'sample_jd.txt'
  sampleResume = 'sample_resume.txt'
  #sampleResume = 'sample_resume.pdf'

  s = Scanner()
  score = s.score(sampleJobDescr, sampleResume)
  print(f'{score:.4f}')

# Entry Point
if __name__ == '__main__':
  main()