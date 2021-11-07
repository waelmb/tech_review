import os
import spacy
import pytextrank  
from datetime import datetime
import yake
import pke

#exact keywords prep
exactKeywordsList = [["SARS-CoV-2", "COVID-19", "Asymptomatic infections" ,"Epidemiological characteristic","Outcome"],
["SARS-CoV-2", "COVID-19", "Virus detection" ,"Diagnostics","Pneumonia", "Serology tests"],
["Remdesivir", "COVID-19", "SARS-Co-V-2" ,"Clinical outcome","Mortality"],
["COVID-19", "Multi-systemic" ,"Lungs","Diffuse alveolar damage"],
["COVID19", "Pneumonia", "Seasonal flu", "The GC/MS data", "Pneumonia and severe seasonalful", "Health Organization (WHO)"],
["Meniere's disease","Meniere disease", "Intratympanic steroid", "Vestibular evoked myogenic potential", "Saccule", "Endolymphatic hydrops"],
["Meniere's disease","Meniere disease", "Intratympanic steroid", "Intratympanic gentamicin", "Endolymphatic hydrops", "Vestibular migraine", "Treatment of Meniere's disease", "Endolymphatic sac shunt"],
["Meniere's disease","Meniere disease", "Endolymphatic hydrops", "Low sodium intake", "Gluten free diet", "Allergy"],
["Downbeat nystagmu", "Meniere disease", "Meniere's disease", "Vertigo crisis"],
["Magnetic resonance imaging", "Inner ear", "Intravenous injection", "Perilymph"]]
exactKeywords = []
for ekl in exactKeywordsList:
    eklset = set()
    for word in ekl:
        for w in word.split(" "):
            eklset.add(w)
        #eklset.add(word)
    exactKeywords.append(eklset)
numOfKeywords = 20

#read files function
def readFiles():
    content = []
    for i in range(1,11):
        filePath = os.getcwd() + '\\data\\' + str(i)+'.txt'
        fileContent = open(filePath, encoding='utf-8').read()
        content.append(fileContent)

    return content

#compares two sets of keywords
def compareWords(estimatedKeywords):
    scores = []
    """ for i, exact in enumerate(exactKeywords):
        score = 0
        for word1 in exact:
            for kwSet in estimatedKeywords[i]:
                for word2 in kwSet.split():
                    if word1.lower() == word2.lower():
                        score+=1
        scores.append(score/len(exactKeywordsList[i]))
    return sum(scores)/len(scores)
    """
    for i, exact in enumerate(exactKeywords):
        rvector = exact.union(estimatedKeywords[i]) 
        l1 =[]
        l2 =[]
        c = 0
        for w in rvector:
            if w in exact: l1.append(1) # create a vector
            else: l1.append(0)
            if w in estimatedKeywords[i]: l2.append(1)
            else: l2.append(0)
        # cosine formula 
        for i in range(len(rvector)):
                c+= l1[i]*l2[i]
        cosine = c / float((sum(l1)*sum(l2))**0.5)
        scores.append(cosine)
    return sum(scores)/len(scores)


def getTopKRankedKeywords(keywords, k, scoreIndex):
    topKeywords = []
    for kw in keywords:
        sortedKeywords = sorted(kw, reverse=True, key=lambda x: x[0])[:k]
        sortedSplitKeywords = set()
        for keyword in sortedKeywords:
            for word in keyword[scoreIndex].split(" "):
                sortedSplitKeywords.add(word)
        topKeywords.append(sortedSplitKeywords)
    return topKeywords

def getScores(keywords, scoreIndex):
    keywords = getTopKRankedKeywords(keywords, numOfKeywords, scoreIndex)
    return compareWords(keywords)

content = readFiles()
output = []

""" #TextRank spaCy model web sm 
nlpSpacy = spacy.load("en_core_web_sm")
nlpSpacy.add_pipe("textrank")
keywordsSpacy = []
now = datetime.now()

for doc in content:
    docKeywords = []
    doc = nlpSpacy(doc)
    for p in doc._.phrases:
        docKeywords.append((p.rank,  p.text))
    keywordsSpacy.append(docKeywords)
delta = (datetime.now()-now).total_seconds()
output.append(("TextRank spaCy web_sm", getScores(keywordsSpacy, 1), delta))

#TextRank spaCy model web sm 
nlpSpacy = spacy.load("en_core_web_md")
nlpSpacy.add_pipe("textrank")
keywordsSpacy = []
now = datetime.now()

for doc in content:
    docKeywords = []
    doc = nlpSpacy(doc)
    for p in doc._.phrases:
        docKeywords.append((p.rank,  p.text))
    keywordsSpacy.append(docKeywords)
delta = (datetime.now()-now).total_seconds()
output.append(("TextRank spaCy web_md", getScores(keywordsSpacy, 1), delta))

#TextRank spaCy model web md Topic Rank
nlpSpacy = spacy.load("en_core_web_trf")
nlpSpacy.add_pipe("textrank")
keywordsSpacy = []
now = datetime.now()

for doc in content:
    docKeywords = []
    doc = nlpSpacy(doc)
    for p in doc._.phrases:
        docKeywords.append((p.rank,  p.text))
    keywordsSpacy.append(docKeywords)
delta = (datetime.now()-now).total_seconds()
output.append(("TextRank spaCy web_trf", getScores(keywordsSpacy, 1), delta))

#Biased TextRank spaCy model web sm 
nlpSpacy = spacy.load("en_core_web_sm")
nlpSpacy.add_pipe("biasedtextrank")
keywordsSpacy = []
now = datetime.now()

for doc in content:
    docKeywords = []
    doc = nlpSpacy(doc)
    for p in doc._.phrases:
        docKeywords.append((p.rank,  p.text))
    keywordsSpacy.append(docKeywords)
delta = (datetime.now()-now).total_seconds()
output.append(("Biased TextRank spaCy web_sm", getScores(keywordsSpacy, 1), delta))

#Biased TextRank spaCy model web sm 
nlpSpacy = spacy.load("en_core_web_md")
nlpSpacy.add_pipe("biasedtextrank")
keywordsSpacy = []
now = datetime.now()

for doc in content:
    docKeywords = []
    doc = nlpSpacy(doc)
    for p in doc._.phrases:
        docKeywords.append((p.rank,  p.text))
    keywordsSpacy.append(docKeywords)
delta = (datetime.now()-now).total_seconds()
output.append(("Biased TextRank spaCy web_md", getScores(keywordsSpacy, 1), delta))

#Biased TextRank spaCy model web md Topic Rank
nlpSpacy = spacy.load("en_core_web_trf")
nlpSpacy.add_pipe("biasedtextrank")
keywordsSpacy = []
now = datetime.now()

for doc in content:
    docKeywords = []
    doc = nlpSpacy(doc)
    for p in doc._.phrases:
        docKeywords.append((p.rank,  p.text))
    keywordsSpacy.append(docKeywords)
delta = (datetime.now()-now).total_seconds()
output.append(("Biased TextRank spaCy web_trf", getScores(keywordsSpacy, 1), delta)) """

#Yake NLTK 
kw_extractor = yake.KeywordExtractor(n=1, dedupLim=0.8, dedupFunc = 'leve')
keywordsYakeNLTK = []
now = datetime.now()

for doc in content:
    keywordsYakeNLTK.append(kw_extractor.extract_keywords(doc))
delta = (datetime.now()-now).total_seconds()
output.append(("Yake NLTK ", getScores(keywordsYakeNLTK, 0), delta))

""" #TextRank PKE
keywordsTextRankPKE = []
extractor = pke.unsupervised.TextRank()
now = datetime.now()
for doc in content:
    extractor.load_document(input=doc, language='en',)
    extractor.candidate_weighting(window=2,
                              pos={'NOUN', 'PROPN', 'ADJ'},
                              top_percent=0.33)
    keywordsTextRankPKE.append(extractor.get_n_best(n=numOfKeywords,  stemming=False, redundancy_removal=True))
delta = (datetime.now()-now).total_seconds()
output.append(("TextRank PKE", getScores(keywordsTextRankPKE, 0), delta))

keywordsYAKEPKE = []
extractor = pke.unsupervised.YAKE()
now = datetime.now()
for doc in content:
    extractor.load_document(input=doc, language='en',)
    extractor.candidate_selection(n=1)   
    extractor.candidate_weighting(window=2)
    keywordsYAKEPKE.append(extractor.get_n_best(n=numOfKeywords,  stemming=False, redundancy_removal=True))
delta = (datetime.now()-now).total_seconds()
output.append(("YAKE PKE", getScores(keywordsYAKEPKE, 0), delta))  """

#done
print(sorted(output, key=lambda x: (x[1], x[2])))