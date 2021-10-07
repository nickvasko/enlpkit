## eNLPKit: An Efficient NLP Toolkit for processing large datasets

eNLPKit is a natural language processing pipeline built on top of [Trankit](https://github.com/nlp-uoregon/trankit), a light-weight
Transformer-based Multilingual NLP toolkit built in PyTorch. This toolkit was built with 
the intended use to be processing very large datasets through batch processing,
so that the processed data can be used in other NLP modeling.

eNLPKit contributes two main implementation upgrades by subclassing the Trankit Pipeline class.
* PyTorch Dataset class that focuses on tokenizing a list of paragraphs, and drops the 
CONLU formatting. This allows for much more efficient preprocessing and batch processing
during token classification.
* eNLPPipeline: is subclass of the Trankit Pipeline, therefore, can process any 
language implemented in Trankit. The subclass implements batch processing 
over the data for the capabilities outlined below. The key difference is that eNLPKit
enables you to pass a list of paragraphs, while still retaining the paragraph structure.
Where Trankit requires you to concatenate all paragraphs, separated by '\n\n', which
losses paragraph structure during processing.

eNLPPipeline Capabilities:

| Task                  | Status   |
|-----------------------|----------|
| Token Classification  | COMPLETE |
| POS Taggering         | NOT IMPLEMENTED |
| Morphological Tagging | NOT IMPLEMENTED |
| Dependency Parsing    | NOT IMPLEMENTED |
| Lemmatization         | NOT IMPLEMENTED |
| MWT Expansion         | NOT IMPLEMENTED |
| NER Tagging           | NOT IMPLEMENTED |

For information on the models used, I'd direct you to the [Trankit technical paper](https://arxiv.org/pdf/2101.03289.pdf).

## Installation

eNLPKit can be easily installed via one of the following methods:

#### From source

```shell script
git clone https://github.com/nickvasko/eNLPKit.git
cd enlpkit
pip install -e .
```

This would first clone our github repo and install eNLPKit.

## Usage

```python
from enlpkit import eNLPPipeline

# a full list of arguments can be found in Trankit documentation
epipe = eNLPPipeline(lang='english', gpu=False, cache_dir='./cache')

docs = ['Hello, World', 'This was processed in batches.']

processed_docs = epipe(docs)

for key in processed_docs:
    print(key)
    print(processed_docs[key])
```

A full example of using eNLPKit to process the 
[WikiText-2 dataset](https://torchtext.readthedocs.io/en/latest/datasets.html#torchtext.datasets.LanguageModelingDataset) 
can be found in the examples folder.

## Acknowledgements

> [1] [Minh Nguyen, Viet Lai, Amir Pouran Ben Veyseh, and Thien Huu Nguyen. 2021. Trankit: A light-weight transformer-based toolkit for multilingual natural language processing. In *Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: System Demonstrations.*](https://aclanthology.org/2021.eacl-demos.10.pdf)

> [2] [Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. 2016. Pointer sentinel mixture models. CoRR, abs/1609.07843.](https://arxiv.org/abs/1609.07843)
