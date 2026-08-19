import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


#given a player_index, returns n player indices that are most similar to the player 
def find_similar(plyr_index, vectors, n):
    try:
        player_vec = vectors.loc[plyr_index].values.reshape(1, -1)
        similarities = cosine_similarity(vectors, player_vec).flatten() #cosine_similarity returns a 2D array of results
        results = pd.Series(
            similarities,
            index=vectors.index,
            name="similarity"
        )
        results = results.drop(index=plyr_index)
        results = results.sort_values(ascending=False)
        return results.index[:n]
    except KeyError:
        raise ValueError(f'Player ID {plyr_index} not found in provided vector data')


def main():
    outfield_vectors = pd.read_csv("../data/clean/outfield_vectors.csv")
    master_players = pd.read_csv("../data/clean/players.csv")
    HAALAND_INDEX = 236
    #should find the five most similar players to Haaland
    sim_indices = find_similar(HAALAND_INDEX, outfield_vectors, 5)
    player_names = [master_players.loc[index].values[1] for index in sim_indices]
    print(player_names)

if __name__ == "__main__":main()
