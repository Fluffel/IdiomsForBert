from transformers import BertModel, BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import torch
import pandas as pd


def get_sentence_key_from_idx(idx):
    return 'sentence' + str(idx)

def load_sentences(file_name):
    data = defaultdict(dict)
    with open(file_name) as file:
        next_line = next(file)
        sample_count = 0
        while next_line:
            sample_key = "sample" + str(sample_count)
            for sentence in range(4):
                data[sample_key][get_sentence_key_from_idx(sentence)] = next_line.rstrip()
                try:
                    next_line = next(file)
                except StopIteration:
                    next_line = None
            sample_count += 1
    return dict(data)

def load_sentences_with_names(file_name):
    def get_next_line(file):
        next_line = next(file).rstrip()
        while next_line == "":
            next_line = next(file).rstrip()
        return next_line

    data = defaultdict(dict)
    with open(file_name) as file:
        next_line = get_next_line(file)
        sample_count = 0
        while next_line:
            sample_key = next_line
            next_line = get_next_line(file)
            for sentence in range(4):
                data[sample_key][get_sentence_key_from_idx(sentence)] = next_line.rstrip()
                try:
                    next_line = get_next_line(file)
                except StopIteration:
                    next_line = None
            sample_count += 1
    return dict(data)
def write_dataframe_to_json(dataframe, file):
    with open(file, '+w') as f:
        f.write(dataframe.to_json(orient='split'))

def get_sample_similarity_score(model, tokenizer, sentences):
    sentence_encodings = {}
    with torch.no_grad():
        for sentence_idx in range(4):
            encoded_input = tokenizer(sentences[get_sentence_key_from_idx(sentence_idx)], add_special_tokens=True, return_tensors='pt')
            output = model(**encoded_input)
            sentence_encodings[sentence_idx] = output
    cosine_similarity_non_idiomatic_meaning = cosine_similarity(sentence_encodings[2]['pooler_output'], sentence_encodings[3]['pooler_output'])
    cosine_similarity_idiomatic_meaning = cosine_similarity(sentence_encodings[0]['pooler_output'], sentence_encodings[1]['pooler_output'])
    similarity_score = cosine_similarity_non_idiomatic_meaning - cosine_similarity_idiomatic_meaning
    return similarity_score.item()


if __name__ == '__main__':
    print("Loading model...")
    model = BertModel.from_pretrained("bert-base-uncased")
    print("Loading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    data = load_sentences_with_names("sentence_quintuples.txt")
    for sample in data.keys():
        score = get_sample_similarity_score(model, tokenizer, data[sample])
        data[sample]['similarity score'] = score

    dataframe = pd.DataFrame.from_dict(data)
    write_dataframe_to_json(dataframe, "similarity_scores.json")
    