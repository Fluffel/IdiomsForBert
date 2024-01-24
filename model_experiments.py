from transformers import BertModel, BertTokenizer
from collections import defaultdict
import pandas as pd

from utils import *


def create_token_diff_avg_0_total_df(sentence_encodings, tokenizer, data, filename):
    print("create dataframe for different tokens average...")
    for sample in data.keys():
        embeddings = get_different_tokens_average_embedding(sentence_encodings[sample], tokenizer, data[sample])
        score, idiomatic_score, non_idiomatic_score = get_sentence_similarity_scores(embeddings)
        data[sample]['interaction score'] = score
        data[sample]['idiomatic score'] = idiomatic_score
        data[sample]['non idiomatic score'] = non_idiomatic_score

    dataframe = pd.DataFrame.from_dict(data)
    write_dataframe_to_json(dataframe, filename)

def create_pool_total_df(sentence_encodings, tokenizer, data, filename):
    print("create dataframe for pooling...")
    for sample in data.keys():
        embeddings = get_pooling_embedding(sentence_encodings[sample])
        score, idiomatic_score, non_idiomatic_score = get_sentence_similarity_scores(embeddings)
        data[sample]['interaction score'] = score
        data[sample]['idiomatic score'] = idiomatic_score
        data[sample]['non idiomatic score'] = non_idiomatic_score

    dataframe = pd.DataFrame.from_dict(data)
    write_dataframe_to_json(dataframe, filename)

def create_len_score_different_avg_df(sentence_encodings, tokenizer, data, filename):
    print("create dataframe for length vs score fo different tokens average...")
    count = 0
    lengths_vs_embedding_val = {'idiomatic': defaultdict(dict), 'non idiomatic': defaultdict(dict)}
    for sample in data.keys():
        sentence_pair_sim_lens = get_sentence_sim_length(tokenizer, data[sample])
        embeddings = get_different_tokens_average_embedding(sentence_encodings[sample], tokenizer, data[sample])

        _, idiomatic_score, non_idiomatic_score = get_sentence_similarity_scores(embeddings)

        lengths_vs_embedding_val['idiomatic']['score'][count]= idiomatic_score
        lengths_vs_embedding_val['idiomatic']['sim length'][count] = sentence_pair_sim_lens[0]
        count += 1
        lengths_vs_embedding_val['non idiomatic']['score'][count] = non_idiomatic_score
        lengths_vs_embedding_val['non idiomatic']['sim length'][count] = sentence_pair_sim_lens[1]
        count += 1

    dataframe = pd.DataFrame.from_dict(lengths_vs_embedding_val)
    print(dataframe)
    write_dataframe_to_json(dataframe, filename)

def create_len_score_pool_df(sentence_encodings, tokenizer, data, filename):
    print("create dataframe for length vs score for pooling...")
    count = 0
    lengths_vs_embedding_val = {'idiomatic': defaultdict(dict), 'non idiomatic': defaultdict(dict)}
    for sample in data.keys():
        sentence_pair_sim_lens = get_sentence_sim_length(tokenizer, data[sample])
        embeddings = get_pooling_embedding(sentence_encodings[sample])

        _, idiomatic_score, non_idiomatic_score = get_sentence_similarity_scores(embeddings)

        lengths_vs_embedding_val['idiomatic']['score'][count]= idiomatic_score
        lengths_vs_embedding_val['idiomatic']['sim length'][count] = sentence_pair_sim_lens[0]
        count += 1
        lengths_vs_embedding_val['non idiomatic']['score'][count] = non_idiomatic_score
        lengths_vs_embedding_val['non idiomatic']['sim length'][count] = sentence_pair_sim_lens[1]
        count += 1

    dataframe = pd.DataFrame.from_dict(lengths_vs_embedding_val)
    write_dataframe_to_json(dataframe, filename)


if __name__ == '__main__':
    print("Loading model...")
    model = BertModel.from_pretrained("bert-base-uncased")
    print("Loading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    data = load_sentences_with_names("data/sentences_revised_2.txt")
    encodings = {}
    print("Encode data...")
    for sample in data.keys():
        encodings[sample] = get_sentence_encodings(model, tokenizer, data[sample])
    
    print("---------------Create Dataframes-----------")
    create_pool_total_df(encodings, tokenizer, data, "data_exports/similarity_scores_pool_revised_2.json")
    # create_token_diff_avg_0_total_df(encodings, tokenizer, data, "data_exports/similarity_scores_diff_0_avg.json")
    # create_len_score_different_avg_df(encodings, tokenizer, data, "data_exports/len_vs_score_diff_0_avg.json")
    # create_len_score_pool_df(encodings, tokenizer, data, "data_exports/len_vs_score_pool.json")
