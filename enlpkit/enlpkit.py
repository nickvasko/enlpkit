from trankit import Pipeline
from trankit.utils.tbinfo import tbname2max_input_length, lang2treebank, tbname2tokbatchsize
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm


class eNLPDataset(Dataset):
    def __init__(self, config, docs, max_input_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(config.embedding_name)
        self.max_input_length = max_input_length
        self.docs = docs

        self.data = self.tokenizer(self.docs, truncation=True, max_length=self.max_input_length,
                                   stride=self.max_input_length-3, return_overflowing_tokens=True,
                                   return_offsets_mapping=True, padding=False)
        self.data['index'] = self.data['overflow_to_sample_mapping']
        self.data['wp_ends'] = [[y[1] for y in x] for x in self.data['offset_mapping']]
        del self.data['overflow_to_sample_mapping']
        del self.data['offset_mapping']

    def __len__(self):
        return len(self.data['input_ids'])

    def __getitem__(self, item):
        return {key: self.data[key][item] for key in self.data}

    def collate_fn(self, batch):
        max_length = max([len(item['input_ids']) for item in batch])
        outputs = {key: [] for key in self.data}
        for item in batch:
            pad_length = max_length - len(item['input_ids'])
            outputs['input_ids'].append(torch.tensor(item['input_ids'] + [self.tokenizer.pad_token_id] * pad_length))
            outputs['attention_mask'].append(torch.tensor([1]*len(item['input_ids']) + [0]*pad_length))
            outputs['index'].append(item['index'])
            outputs['wp_ends'].append(torch.tensor(item['wp_ends'] + [0] * pad_length))
        return {key: torch.stack(outputs[key]) if type(outputs[key][0]) == torch.Tensor else outputs[key] for key in outputs}


class eNLPPipeline(Pipeline):
    def _tokenize_docs(self, in_docs):
        eval_batch_size = tbname2tokbatchsize.get(lang2treebank[self.active_lang], self._tokbatchsize)

        dataset = eNLPDataset(self._config, in_docs,
                              max_input_length=tbname2max_input_length.get(lang2treebank[self.active_lang], 400))

        self._load_adapter_weights(model_name='tokenizer')

        dataloader = DataLoader(dataset, batch_size=eval_batch_size, shuffle=False, collate_fn=dataset.collate_fn)
        outputs = {key: [] for key in ['index', 'token_labels', 'wp_ends']}
        for batch in tqdm(dataloader):
            batch['input_ids'] = batch['input_ids'].to(self._config.device)
            batch['attention_mask'] = batch['attention_mask'].to(self._config.device)

            wordpiece_reprs = self._embedding_layers.encode(piece_idxs=batch['input_ids'],
                                                            attention_masks=batch['attention_mask'])
            wordpiece_scores = self._tokenizer[self._config.active_lang].tokenizer_ffn(wordpiece_reprs)
            batch['token_labels'] = torch.argmax(wordpiece_scores, dim=2).cpu()
            labels, ends = [], []
            for l, e in zip(batch['token_labels'], batch['wp_ends']):
                i = e.argmax()
                labels.append(l[:i].tolist())
                ends.append(e[1:i + 1].tolist())
            outputs['index'].extend(batch['index'])
            outputs['token_labels'].extend(labels)
            outputs['wp_ends'].extend(ends)

        return outputs

    def __call__(self, inputs):
        tokenized_docs = self._tokenize_docs(in_docs=inputs)
        return tokenized_docs
