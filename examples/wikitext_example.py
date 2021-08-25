from enlpkit.enlpkit import eNLPPipeline
from torchtext.datasets import WikiText2

epipe = eNLPPipeline(lang='english', gpu=False, cache_dir='./cache')
epipe._tokbatchsize = 16

wikidataset = WikiText2()
wikidata = [x for d in wikidataset for x in d if x != ' \n']
print('# examples: ', len(wikidata))

processed_docs = epipe(wikidata)
