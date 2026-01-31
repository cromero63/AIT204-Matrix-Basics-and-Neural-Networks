import pandas as pd
import torch
from nba_team_selectorANN import NbaTeamSelectorANN
import json
def Main():
    # Read data in and list IDs of Optimal players
    nbaPlayerSet = pd.read_csv("../data/all_seasons.csv")
    optimal_players = [12520, 10634, 12332, 11781, 11670] # Optimal 5

    # Create the pool of 100 players ensure our 5 optimal are included
    optimal_players = nbaPlayerSet[nbaPlayerSet['id'].isin(optimal_players)]
    other_players = nbaPlayerSet[~nbaPlayerSet['id'].isin(optimal_players)].sample(n=95, random_state=42)
    
    # combine and shuffle so model doesnt learn placement
    pool_100 = pd.concat([optimal_players, other_players]).sample(frac=1).reset_index(drop=True)

    # create mapping to make validation easier
    index_to_name = pool_100['player_name'].to_dict()
    index_to_id = pool_100['id'].to_dict()

    # Map our optimal_players to a target column
    pool_100['target'] = pool_100['id'].isin(optimal_players).astype(float)

    # feature columns that we are using
    feature_cols = ["player_height", "player_weight", "gp", "pts", "reb", "ast", 
                    "net_rating", "oreb_pct", "dreb_pct", "usg_pct", "ts_pct", "ast_pct"]
    
    # Flatten X and y
    X = pool_100[feature_cols].values.flatten().reshape(1, -1)
    y = pool_100["target"].values.reshape(1, -1)

    # Create and train model, 100 output columsn as we have 100 players
    model = NbaTeamSelectorANN(input_dim=len(feature_cols)*100, num_classes=100)
    model.train_model(X, y, 100)

    # Get the one-hot vector and the indices of the top 5
    one_hot, top_indices = model.predict(X)
    
    # Flatten top_indices to a simple list
    selected_indices = top_indices.flatten()
    
    # Map back to the dataframe
    picked_ids = [index_to_id[i] for i in selected_indices]
    picked_names = [index_to_name[i] for i in selected_indices]

    print("--- THE ML STARTING FIVE ---")
    for name, pid in zip(picked_names, picked_ids):
        print(f"Player: {name} (ID: {pid})")

    # Validation logic using ID
    # We check if the ID of the picked player is in our original 'optimal_ids' list
    correct_picks = [pid for pid in picked_ids if pid in optimal_players]
    print(f"\nAI correctly identified {len(correct_picks)}/5 designated optimal IDs.")
    if correct_picks:
        print(f"Correct IDs: {correct_picks}")

    # Export artifacts
    torch.save(model.state_dict(), "../data/model_state_dict.pt")
    with open("../data/label_names.json","w") as f: json.dump(feature_cols, f)
    print("Exported: model_state_dict.pt, label_names.json")


Main()