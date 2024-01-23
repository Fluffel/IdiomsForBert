import pandas as pd
import matplotlib.pyplot as plt
import textwrap



def print_len_vs_score():
    f = open("data_exports/len_vs_score.json")
    df = pd.read_json(f, orient='split')
    # print(df)
    df.plot.scatter('sim length', 'score')
    plt.show()

def print_len_vs_score_fract():
    f = open("data_exports/len_vs_score_fract.json")
    df = pd.read_json(f, orient='split')
    # file = open("data_exports/df_print", "w")
    # file.write(df.to_string())
    df.plot.scatter('sim length', 'score')
    plt.show()


def print_similarity_scores():
    sim_pooling = open("data_exports/similarity_scores_pooling.json")
    sim_tokens_avg = open("data_exports/similarity_scores_token_avg.json")
    # with open("similarity_scores_pooling.json") as f:
    df_pool = pd.read_json(sim_pooling, orient='split')
    df_token = pd.read_json(sim_tokens_avg, orient='split')
    
    interaction_scores_pool = df_pool.loc["interaction score"]
    interaction_scores_token = df_token.loc["interaction score"]
    # idiomatic_scores = df.loc["idiomatic score"] - 1
    # non_idiomatic_score = df.loc["non idiomatic score"]

    x_labels = df_token.columns
    # scores = scores.transpose()

    # # Plotting the bar chart
    interaction_scores_pool.plot(kind='bar', legend=False, figsize=(12, 10))
    # idiomatic_scores.plot(kind='bar', legend=False, figsize=(12,8))
    wrapped_labels = [textwrap.fill(label, width=10) for label in x_labels]
    plt.title('Similarity Scores for Sentences using pooling method')
    plt.ylabel('Similarity Score')
    # plt.xticks(range(len(wrapped_labels)), wrapped_labels, rotation=45)
    plt.savefig('similarity_scores_pool.png')

    interaction_scores_token.plot(kind='bar', legend=False, figsize=(12, 10))
    plt.title('Similarity Scores for Sentences using only different tokens')
    plt.ylabel('Similarity Score')
    # plt.xticks(range(len(wrapped_labels)), wrapped_labels, rotation=45)
    plt.savefig('similarity_scores_tokens.png')

if __name__ == '__main__':
    print_len_vs_score_fract()

