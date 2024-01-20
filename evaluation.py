import pandas as pd
import matplotlib.pyplot as plt
import textwrap



if __name__ == '__main__':
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
    plt.xticks(range(len(wrapped_labels)), wrapped_labels, rotation=45)
    plt.savefig('similarity_scores_pool.png')

    interaction_scores_token.plot(kind='bar', legend=False, figsize=(12, 10))
    plt.title('Similarity Scores for Sentences using only different tokens')
    plt.ylabel('Similarity Score')
    plt.xticks(range(len(wrapped_labels)), wrapped_labels, rotation=45)
    plt.savefig('similarity_scores_tokens.png')


# import pandas as pd
# import matplotlib.pyplot as plt
# import textwrap

# # Open the JSON files
# sim_pooling = open("data_exports/similarity_scores_pooling.json")
# sim_tokens_avg = open("data_exports/similarity_scores_tokens.json")

# # Read the JSON files into DataFrames
# df_pool = pd.read_json(sim_pooling, orient='split')
# df_token = pd.read_json(sim_tokens_avg, orient='split')

# # Extract interaction scores from DataFrames
# interaction_scores_pool = df_pool.loc["interaction score"]
# interaction_scores_token = df_token.loc["interaction score"]

# # Extract x-axis labels
# x_labels = df_pool.columns

# # Plotting the bar charts side by side
# fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 8))

# # Plot for similarity_scores_only_pool.png
# axes[0].bar(range(len(x_labels)), interaction_scores_pool, color='blue')
# axes[0].set_title('Similarity Scores for Sentences (Pooling)')
# axes[0].set_ylabel('Similarity Score')
# axes[0].set_xticks(range(len(x_labels)))
# axes[0].set_xticklabels([textwrap.fill(label, width=10) for label in x_labels], rotation=45)

# # Plot for similarity_scores_only_tokens.png
# axes[1].bar(range(len(x_labels)), interaction_scores_token, color='green')
# axes[1].set_title('Similarity Scores for Sentences (Tokens)')
# axes[1].set_ylabel('Similarity Score')
# axes[1].set_xticks(range(len(x_labels)))
# axes[1].set_xticklabels([textwrap.fill(label, width=10) for label in x_labels], rotation=45)

# # Adjust layout and save the plot
# plt.tight_layout()
# plt.savefig('evaluation_exports/similarity_scores_combined.png')
# plt.show()
