from transformers import BertModel, BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import torch



def load_sentences(file_name):
    data = defaultdict(dict)
    with open(file_name) as file:
        next_line = next(file)
        sample_count = 0
        while next_line:
            sample_key = "sample" + str(sample_count)
            for sentence in range(4):
                data[sample_key][sentence] = next_line
                try:
                    next_line = next(file)
                except StopIteration:
                    next_line = None
            sample_count += 1
    return data


def get_sample_similarity_score(model, tokenizer, sentences):
    sentence_encodings = {}
    with torch.no_grad():
        for sentence_idx in range(4):
            encoded_input = tokenizer(sentences[sentence_idx], add_special_tokens=True, return_tensors='pt')
            output = model(**encoded_input)
            sentence_encodings[sentence_idx] = output
    cosine_similarity_non_idiomatic_meaning = cosine_similarity(sentence_encodings[2]['pooler_output'], sentence_encodings[3]['pooler_output'])
    cosine_similarity_idiomatic_meaning = cosine_similarity(sentence_encodings[0]['pooler_output'], sentence_encodings[1]['pooler_output'])
    similarity_score = cosine_similarity_non_idiomatic_meaning - cosine_similarity_idiomatic_meaning
    return similarity_score


if __name__ == '__main__':
    print("Loading model...")
    model = BertModel.from_pretrained("bert-base-uncased")
    print("Loading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    data = load_sentences("sentences.txt")
    for sample in data.keys():
        score = get_sample_similarity_score(model, tokenizer, data[sample])
        data[sample]['score'] = score
    