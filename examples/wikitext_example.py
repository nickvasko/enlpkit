from enlpkit.enlpkit import eNLPPipeline
from torchtext.datasets import WikiText2

epipe = eNLPPipeline(lang='english', gpu=False, cache_dir='./cache')

wikidataset = WikiText2()
wikidata = [x for d in wikidataset for x in d if x != ' \n']
print('# examples: ', len(wikidata))

processed_docs = epipe(wikidata)

for key in processed_docs:
    print(key)
    print(processed_docs[key])
