import pandas as pd
import matplotlib.pyplot as plt
import json
import textwrap

def load_sentence_2_dic():
    with open("data_exports/sample_2_idx.json") as f:
        return json.load(f)

def print_len_vs_score():
    f = open("data_exports/len_vs_score.json")
    df = pd.read_json(f, orient='split')
    # print(df)
    df.plot.scatter('sim length', 'score')
    plt.savefig('evaluation_exports/len_vs_score_pool.png')

def print_len_vs_score_fract():
    f = open("data_exports/len_vs_score_fract.json")
    df = pd.read_json(f, orient='split')
    # file = open("data_exports/df_print", "w")
    # file.write(df.to_string())
    df.plot.scatter('sim length', 'score')
    plt.savefig('evaluation_exports/len_vs_score_diff_tokens_fract.png')

def print_len_vs_score_avg():
    f = open("data_exports/len_vs_score_avg.json")
    df = pd.read_json(f, orient='split')
    # print(df)
    df.plot.scatter('sim length', 'score')
    plt.savefig('evaluation_exports/len_vs_score_diff_tokens.png')

def print_similarity_scores_pooling():
    sim_pooling = open("data_exports/similarity_scores_pooling.json")
    df_pool = pd.read_json(sim_pooling, orient='split')
    
    interaction_scores_pool = df_pool.loc["interaction score"].values
    x_labels = [i for i in range(len(df_pool.columns))]

    # # Plotting the bar chart
    plt.figure(figsize=(10,6))
    plt.bar(x_labels, interaction_scores_pool)
    # interaction_scores_pool.plot(kind='bar', legend=False, figsize=(12, 10))
    # idiomatic_scores.plot(kind='bar', legend=False, figsize=(12,8))
    # wrapped_labels = [textwrap.fill(label, width=10) for label in x_labels]
    plt.title('Similarity Scores for Sentences using pooling method')
    plt.ylabel('Similarity Score')
    plt.xlabel('Numbered Idioms')
    # plt.xticks(range(len(wrapped_labels)), wrapped_labels, rotation=45)
    plt.savefig('evaluation_exports/similarity_scores_pool.png')

def print_similarity_scores_diff_avg():
    sim_tokens_avg = open("data_exports/similarity_scores_diff_tokens_avg.json")
    df_token = pd.read_json(sim_tokens_avg, orient='split')
    interaction_scores_token = df_token.loc["interaction score"].values
    # x_labels = df_token.columns
    x_labels = [i for i in range(len(df_token.columns))]
    plt.figure(figsize=(10,6))
    plt.bar(x_labels, interaction_scores_token)
    plt.title('Similarity Scores for Sentences using the average of only different tokens')
    plt.ylabel('Similarity Score')
    plt.xlabel('Idioms')
    # plt.xticks(range(len(wrapped_labels)), wrapped_labels, rotation=45)
    plt.savefig('evaluation_exports/similarity_scores_diff_tokens_avg.png')

def print_i_non_i_scatter_pool():
    sim_pooling = open("data_exports/similarity_scores_pooling.json")
    df_pool = pd.read_json(sim_pooling, orient='split')
    scores_idiomatic_pool = df_pool.loc["idiomatic score"].values
    scores_non_idiomatic_pool = df_pool.loc["non idiomatic score"].values

    x_labels = [i for i in range(len(df_pool.columns))]
    plt.figure(figsize=(15,6))
    plt.scatter(x_labels, scores_idiomatic_pool, label='Idiomatic', marker='o', color='green')
    plt.scatter(x_labels, scores_non_idiomatic_pool, label='Non Idiomatic', marker='o',color='red')

    plt.title('Scatter Plot of Idiomatic and Non-Idiomatic Scores')
    plt.xlabel("Numbered Idioms")
    plt.ylabel('Cosine Similarity of sentence pairs')
    plt.legend()
    plt.savefig('evaluation_exports/scatter_both_scores_pool.png')
if __name__ == '__main__':
    # print_len_vs_score()
    # print_similarity_scores_diff_avg()
    # print_len_vs_score()
    # print_len_vs_score_avg()
    # print_len_vs_score_fract()
    data = load_sentence_2_dic()
    print(data)

