from enlpkit.enlpkit import eNLPPipeline

# a full list of arguments can be found in Trankit documentation
epipe = eNLPPipeline(lang='english', gpu=False, cache_dir='./cache')

docs = ['Hello, World', 'This was processed in batches.']

processed_docs = epipe(docs)

for key in processed_docs:
    print(key)
    print(processed_docs[key])
