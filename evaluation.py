import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
import textwrap

def load_sentence_2_dic():
    with open("data_exports/sample_2_idx.json") as f:
        return json.load(f)

def print_revised_vs_old(source, save):
    with open("data_exports/" + "similarity_scores_pool.json") as f:
        df_pool = pd.read_json(f, orient='split')
    with open("data_exports/" + source) as f:
        df_pool_r = pd.read_json(f, orient='split')
    
    width = 0.3
    x_labels = df_pool_r.columns
    wrapped_labels = [textwrap.fill(label, width=10) for label in x_labels]
    ind = np.arange(len(x_labels))
    df_pool_values = df_pool[x_labels].loc['interaction score'].values
    df_pool_r_values = df_pool_r.loc['interaction score'].values

    plt.figure(figsize=(12,10))

    plt.bar(ind, df_pool_values, width, label='Original')
    plt.bar(ind + width, df_pool_r_values, width, label='Revised')
    plt.xticks(ind + width / 2, wrapped_labels, rotation=45)

    plt.xlabel('Sample Idiom')
    plt.ylabel('Interaction Score')
    plt.title('Comparison of Interaction Scores with Revised Sentences')
    plt.legend()
    plt.savefig("evaluation_exports/" + save)


def print_len_vs_score(source, save, alpha, title):
    f = open("data_exports/" + source)
    df = pd.read_json(f, orient='split')
    
    idiom = df['idiomatic']
    n_idiom = df['non idiomatic']
    x_ticks = np.arange(len(idiom.loc['score'].items()))
    plt.scatter(idiom.loc['sim length'].values(), idiom.loc['score'].values(), label='Idiomatic', color='green', alpha=alpha)
    plt.scatter(n_idiom.loc['sim length'].values(), n_idiom.loc['score'].values(), label='Non Idiomatic', color='red', alpha=alpha)

    plt.title(title)
    plt.xlabel("Amount Identical Tokens")
    plt.ylabel('Cosine Similarity of sentence pairs')
    plt.legend()
    plt.savefig('evaluation_exports/' + save)
    f.close()

# It looks like taking the fraction does not solve the issue of any correlation between length and score
def print_len_vs_score_fract():
    f = open("data_exports/len_vs_score_fract.json")
    df = pd.read_json(f, orient='split')
    df.plot.scatter('sim length', 'score')
    plt.savefig('evaluation_exports/len_vs_score_diff_tokens_fract.png')
    f.close()


def print_similarity_scores(source, save, title):
    sim_pooling = open("data_exports/" + source)
    df_pool = pd.read_json(sim_pooling, orient='split')
    
    interaction_scores_pool = df_pool.loc["interaction score"].values
    x_labels = [i for i in range(len(df_pool.columns))]

    # # Plotting the bar chart
    plt.figure(figsize=(10,6))
    plt.bar(x_labels, interaction_scores_pool)
    plt.title(title)
    plt.ylabel('Similarity Score')
    plt.xlabel('Numbered Idioms')
    # plt.xticks(range(len(wrapped_labels)), wrapped_labels, rotation=45)
    plt.savefig('evaluation_exports/' + save)

# def print_similarity_scores_diff_avg():
#     sim_tokens_avg = open("data_exports/similarity_scores_diff_tokens_avg.json")
#     df_token = pd.read_json(sim_tokens_avg, orient='split')
#     interaction_scores_token = df_token.loc["interaction score"].values
#     # x_labels = df_token.columns
#     x_labels = [i for i in range(len(df_token.columns))]
#     plt.figure(figsize=(10,6))
#     plt.bar(x_labels, interaction_scores_token)
#     plt.title('Similarity Scores for Sentences using the average of only different tokens')
#     plt.ylabel('Similarity Score')
#     plt.xlabel('Idioms')
#     # plt.xticks(range(len(wrapped_labels)), wrapped_labels, rotation=45)
#     plt.savefig('evaluation_exports/similarity_scores_diff_tokens_avg.png')

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

def print_similarity_scores_both(source_i, source_ni, save, title):
    with open("data_exports/" + source_i) as f:
        df_pool = pd.read_json(f, orient='split')
    with open("data_exports/" + source_ni) as f:
        df_diff = pd.read_json(f, orient='split')

    interaction_scores_pool = df_pool.loc["interaction score"].values
    interaction_scores_diff = df_diff.loc["interaction score"].values
    scale = sum(abs(interaction_scores_pool))/sum(abs(interaction_scores_diff))
    interaction_scores_diff *= scale
    # x_labels = [i for i in range(len(df_pool.columns))]
    x_labels = np.arange(len(df_pool.columns))

    color_pool = 'green'
    color_diff = 'red'
    a = 0.8
    width = 0.5
    # wrapped_labels = [textwrap.fill(label, width=10) for label in x_labels]
    # df_pool_values = df_pool[x_labels].loc['interaction score'].values
    # df_pool_r_values = df_pool_r.loc['interaction score'].values

    plt.figure(figsize=(15,10))

    plt.bar(x_labels, interaction_scores_pool, width, label='Pooling', color=color_pool, alpha=a)
    plt.bar(x_labels + width, interaction_scores_diff, width, label='Scaled Different Token Average', color=color_diff, alpha=a)

    plt.axhline(y=np.mean(interaction_scores_pool), color=color_pool, linestyle='--', label='Overall Mean (Pooling)')
    plt.axhline(y=np.mean(interaction_scores_diff), color=color_diff, linestyle='--', label='Overall Mean (Diff)')

    plt.xticks(x_labels + width / 2, x_labels, rotation=90)
    plt.title(title)
    plt.xlabel("Numbered Idioms")
    plt.ylabel('Cosine Similarity of Sentence Pairs')
    plt.legend()
    # plt.show()
    plt.savefig('evaluation_exports/' + save)

    
if __name__ == '__main__':
    print_similarity_scores_both("similarity_scores_pool.json", "similarity_scores_diff_0_avg.json", "similarity_scores_both.png", "Scores for Pooling vs. Different Token Average")
    # print_len_vs_score("len_vs_score_diff_0_avg.json", "len_vs_score_only_diff_tokens.png", 0.7, "Length-Score Interaction for only Different Tokens")
    # plt.clf()
    # print_len_vs_score("len_vs_score_pool.json", "len_vs_score_pool.png", 0.3, "Length-Score Interaction for Pooling")
    # print_len_vs_score_avg()
    # print_len_vs_score_fract()
    # print_revised_vs_old("similarity_scores_pool_revised_2.json", "comparison_revised_old_simscores_2.png")

