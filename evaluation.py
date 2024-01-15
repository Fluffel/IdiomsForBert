import pandas as pd
import matplotlib.pyplot as plt
import textwrap



if __name__ == '__main__':
    with open("similarity_scores.json") as f:
        df = pd.read_json(f, orient='split')
        # print(df.loc["similarity score"])
        scores = df.loc["similarity score"]
        x_labels = df.columns
        # scores = scores.transpose()

        # # Plotting the bar chart
        scores.plot(kind='bar', legend=False, figsize=(12, 10))
        wrapped_labels = [textwrap.fill(label, width=10) for label in x_labels]
        plt.title('Similarity Scores for Sentences')
        # plt.xlabel('Phrases')
        plt.ylabel('Similarity Score')
        plt.xticks(range(len(wrapped_labels)), wrapped_labels, rotation=45)
        plt.savefig('sim_scores.png')
