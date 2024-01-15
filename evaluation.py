import pandas as pd
import matplotlib.pyplot as plt



if __name__ == '__main__':
    with open("similarity_scores.json") as f:
        df = pd.read_json(f, orient='split')
        print(df)
        # plt.bar(df["similarity score"])
        # plt.show()
