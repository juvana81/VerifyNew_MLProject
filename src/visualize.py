import pandas as pd
import plotly.express as px

def visualize_plotly():
    df = pd.read_csv("data/clustered_media_bias.csv")

    fig = px.scatter(df, x="PCA1", y="PCA2", color=df["Cluster"].astype(str),
                     hover_data=["Source"], title="Media Bias Map")
    fig.show()

if __name__ == "__main__":
    visualize_plotly()
