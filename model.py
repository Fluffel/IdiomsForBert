from transformers import BertModel, BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import torch
import pandas as pd


NUMBER_SENTENCES_PER_SAMPLE = 4
NUMBER_SENTENCE_PAIRS_PER_SAMPLE = 2

def get_sentence_key_from_idx(idx):
    return 'sentence' + str(idx)

def load_sentences(file_name):
    data = defaultdict(dict)
    with open(file_name) as file:
        next_line = next(file)
        sample_count = 0
        while next_line:
            sample_key = "sample" + str(sample_count)
            for sentence in range(NUMBER_SENTENCES_PER_SAMPLE):
                data[sample_key][get_sentence_key_from_idx(sentence)] = next_line.rstrip()
                try:
                    next_line = next(file)
                except StopIteration:
                    next_line = None
            sample_count += 1
    return dict(data)


'''
    returns Dictionary with:
    {idiom: {
        sentence1: "",
        sentence2: "",
        sentence3: "",
        sentence4: ""
    }}
'''
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
            idiom = next_line
            next_line = get_next_line(file)
            for sentence in range(4):
                data[idiom][get_sentence_key_from_idx(sentence)] = next_line.rstrip()
                try:
                    next_line = get_next_line(file)
                except StopIteration:
                    next_line = None
            sample_count += 1
    return dict(data)


def write_dataframe_to_json(dataframe, file):
    with open(file, '+w') as f:
        f.write(dataframe.to_json(orient='split'))

def get_sentence_encodings(model, tokenizer, sentences):
    sentence_encodings = {}
    with torch.no_grad():
        for sentence_idx in range(NUMBER_SENTENCES_PER_SAMPLE):
            encoded_input = tokenizer(sentences[get_sentence_key_from_idx(sentence_idx)], add_special_tokens=True, return_tensors='pt')
            output = model(**encoded_input)
            sentence_encodings[sentence_idx] = output
    return sentence_encodings

def get_different_token_indeces(tokenizer, sentences, offset=1):
    tokenized_sentences = {}
    indeces_different_tokens = {}

    for idx in range(NUMBER_SENTENCES_PER_SAMPLE):
        tokens = tokenizer.tokenize(sentences[get_sentence_key_from_idx(idx)])
        tokenized_sentences[idx] = ['[SPACE]'] * offset + tokens

    for pair in range(NUMBER_SENTENCE_PAIRS_PER_SAMPLE):
        min_identical_range = 0
        max_identical_range = 0 #from the back
        while tokenized_sentences[2*pair][min_identical_range] == tokenized_sentences[2*pair + 1][min_identical_range]:
            min_identical_range +=1
        while tokenized_sentences[2*pair][max_identical_range - 1] == tokenized_sentences[2*pair + 1][max_identical_range - 1]:
            max_identical_range -= 1
        for i in range(2):
            indeces_different_tokens[2*pair + i] = [idx for idx in range(min_identical_range, len(tokenized_sentences[2*pair + i]) + max_identical_range)]
    return indeces_different_tokens

    

def get_different_tokens_average_embedding(sentence_encodings, tokenizer, sentences):
    token_indeces = get_different_token_indeces(tokenizer, sentences)
    sentences_avg = {}
    for i in range(NUMBER_SENTENCES_PER_SAMPLE):
        sentence_encoding = sentence_encodings[i].last_hidden_state.squeeze() 
        token_embeddings = sentence_encoding[token_indeces[i]]
        sentences_avg[i] = torch.mean(token_embeddings, dim=0).unsqueeze(0)
    return sentences_avg
    
def get_pooling_embedding(sentence_encodings):
    for sentence in sentence_encodings.keys():
        sentence_encodings[sentence] = sentence_encodings[sentence].pooler_output
    return sentence_encodings




def get_sentence_similarity_scores(sentence_embeddings):
    cosine_similarity_idiomatic_meaning = cosine_similarity(sentence_embeddings[0], sentence_embeddings[1])
    cosine_similarity_non_idiomatic_meaning = cosine_similarity(sentence_embeddings[2], sentence_embeddings[3])
    similarity_score = cosine_similarity_non_idiomatic_meaning - cosine_similarity_idiomatic_meaning
    return similarity_score.item(), cosine_similarity_idiomatic_meaning.item(), cosine_similarity_non_idiomatic_meaning.item()


if __name__ == '__main__':
    print("Loading model...")
    model = BertModel.from_pretrained("bert-base-uncased")
    print("Loading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    data = load_sentences_with_names("sentence_quintuples_20_1.txt")
    for sample in data.keys():
        sentence_encodings = get_sentence_encodings(model, tokenizer, data[sample])
        # embeddings = get_different_tokens_average_embedding(sentence_encodings, tokenizer, data[sample])
        embeddings = get_pooling_embedding(sentence_encodings)
        score, idiomatic_score, non_idiomatic_score = get_sentence_similarity_scores(embeddings)
        # score, idiomatic_score, non_idiomatic_score = get_sentence_similarity_scores(model, tokenizer, data[sample])
        data[sample]['interaction score'] = score
        data[sample]['idiomatic score'] = idiomatic_score
        data[sample]['non idiomatic score'] = non_idiomatic_score

    dataframe = pd.DataFrame.from_dict(data)
    write_dataframe_to_json(dataframe, "data_exports/similarity_scores_pooling.json")
    