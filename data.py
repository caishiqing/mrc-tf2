from numpy.lib.shape_base import expand_dims
from transformers import AutoTokenizer
from collections import OrderedDict
from typing import List, Union
import numpy as np
import uuid


class DataLoader(object):
    def __init__(self, tokenizer: AutoTokenizer,
                 max_input_length: int = 512,
                 max_answer_length: int = 8,
                 doc_stride: int = 128):

        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_answer_length = max_answer_length
        self.doc_stride = doc_stride

        permut_indicate = np.tri(max_input_length, k=max_answer_length-1) - np.tri(max_input_length, k=-1)
        start_indices, end_indices = np.where(permut_indicate == 1)
        self.start_indices = start_indices.tolist()
        self.end_indices = end_indices.tolist()
        self.indice_map = {(s, e): i for i, (s, e) in enumerate(zip(self.start_indices, self.end_indices))}

    def _pre_process(self, record: dict):
        tokenized_example = self.tokenizer(
            record["question"],
            record["context"],
            truncation="only_second",
            max_length=self.max_input_length,
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        qid = record.get('id', str(uuid.uuid1()))

        examples = []
        for i in range(len(tokenized_example['input_ids'])):
            example = {}
            example['input_ids'] = tokenized_example['input_ids'][i]
            example['token_type_ids'] = tokenized_example['token_type_ids'][i]
            example['attention_mask'] = tokenized_example['attention_mask'][i]
            example['offset_mapping'] = tokenized_example['offset_mapping'][i]

            if record.get('answer'):
                answer = record['answer']
                example['answer'] = answer
                answer_char_start = record.get('answer_start', record['context'].find(answer))
                answer_char_end = record.get('answer_end', answer_char_start + len(answer))
                sequence_ids = tokenized_example.sequence_ids(i)

                token_start_id = 0
                while sequence_ids[token_start_id] != 1:
                    token_start_id += 1

                token_end_id = len(example['input_ids']) - 1
                while sequence_ids[token_end_id] != 1:
                    token_end_id -= 1

                # answer is not completely in sequence
                if answer_char_start < example['offset_mapping'][token_start_id][0] \
                        or answer_char_end > example['offset_mapping'][token_end_id][1]:
                    continue

                answer_token_start = token_start_id
                while answer_token_start <= token_end_id and \
                        example['offset_mapping'][answer_token_start][0] <= answer_char_start:
                    answer_token_start += 1
                example['answer_token_start'] = answer_token_start - 1

                answer_token_end = token_end_id
                while example['offset_mapping'][answer_token_end][1] >= answer_char_end:
                    answer_token_end -= 1
                example['answer_token_end'] = answer_token_end + 1

                # answer length exceed
                if example['answer_token_end']-example['answer_token_start'] >= self.max_answer_length:
                    continue

                example['label'] = self.indice_map[(example['answer_token_start'], example['answer_token_end'])]
            else:
                example['context'] = record['context']
                example['qid'] = qid

            examples.append(example)

        return examples

    def pre_process(self, records: List[dict]):
        all_examples = []
        for record in records:
            examples = self._pre_process(record)
            all_examples += examples

        return all_examples

    def post_process(self, examples: List[dict], predicts: Union[List[list], np.ndarray]):
        argmax = np.argmax(predicts, axis=1)
        scores = np.max(predicts, axis=1)
        results = OrderedDict()

        for index, score, example in zip(argmax, scores, examples):
            token_start = self.start_indices[index]
            token_end = self.end_indices[index]
            offset_mapping = example['offset_mapping']
            char_start = offset_mapping[token_start][0]
            char_end = offset_mapping[token_end][1]
            answer = example['context'][char_start:char_end]
            results.setdefault(example['qid'], []).append((answer, score))

        for key, val in results.items():
            answer, score = max(val, key=lambda x: x[1])
            results[key] = {'id': key, 'prediction': answer, 'score': float(round(score, 4))}

        return results

    def get_dataset(self, examples: List[dict], mode: str = 'train'):
        input_ids = np.zeros(shape=(len(examples), self.max_input_length), dtype=np.int32)
        token_type_ids = np.zeros(shape=(len(examples), self.max_input_length), dtype=np.int32)
        attention_mask = np.zeros(shape=(len(examples), self.max_input_length), dtype=np.int32)
        labels = np.zeros(shape=(len(examples),), dtype=np.int32) if mode == 'train' else None

        for i, example in enumerate(examples):
            input_ids[i] = example['input_ids']
            token_type_ids[i] = example['token_type_ids']
            attention_mask[i] = example['attention_mask']
            if mode == 'train':
                labels[i] = example['label']

        return (input_ids, token_type_ids, attention_mask), labels


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
    data_loader = DataLoader(tokenizer, 20, doc_stride=5)

    example = {
        'qid': '123456',
        'question': '姚明有多高？',
        'context': '姚明身高226cm，被称为小巨人。',
        'answer': '226cm',
        'answer_start': 4
    }
    examples = data_loader._pre_process(example)

    preds = np.zeros(shape=(len(example), len(data_loader.indice_map)))
    for i in range(len(examples)):
        examples[i]['context'] = example['context']
        examples[i]['qid'] = example['qid']
        preds[i][examples[i]['label']] = 1

    print(examples)
    print(data_loader.post_process(examples, preds))
