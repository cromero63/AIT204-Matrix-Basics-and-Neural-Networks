import streamlit as st
import pandas as pd
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from backend.nba_team_selectorANN import NbaTeamSelectorANN

def Main():
    st.set_page_config(
        page_title="Matrix Basics and Neural Networks",
        layout="wide"
    )

    st.title("Matrix Basics and Neural Networks")

    st.header("Part 1")
    st.markdown("""
        a. How do you represent an $a_{i,j}$ matrix?

        &emsp; An $a_{i,j}$ matrix represents a matrix with $i$ rows and $j$ columns.


        b. Represent a row vector and a column vector.

        &emsp;**Row vector:** a matrix with only one row.

        &emsp;&emsp; $R = [r_1, r_2, ..., r_n]$

        &emsp;**Column vector:** a matrix with only one column.

        &emsp;&emsp; $C = \\begin{bmatrix} c_1 \\\\ c_2 \\\\ \\vdots \\\\ c_m \\end{bmatrix}$

        c. Let $A = \\begin{bmatrix} 1 & -2 & 3 \\\\ 4 & 5 & -6 \\end{bmatrix}$ and $B = \\begin{bmatrix} 3 & 0 & 2 \\\\ -7 & 1 & 8 \\end{bmatrix}$ find:

        1. $A + B$

        &emsp; $\\begin{bmatrix} 4 & -2 & 5 \\\\ -3 & 6 & 2 \\end{bmatrix}$

        2. $3A$

        &emsp; $\\begin{bmatrix} 3 & -6 & 9 \\\\ 12 & 15 & -18 \\end{bmatrix}$

        3. $2A - 3B$

        &emsp; $\\begin{bmatrix} -7 & -4 & 0 \\\\ 29 & 7 & -36 \\end{bmatrix}$

        4. $AB$

            Not defined. AB is not possible, because matrix multiplication requires the number of columns in the first matrix to equal the number of columns in the second matrix. In this case both matrices are 2x3.

        5. $A^T$

        &emsp; $\\begin{bmatrix} 1 & 4 \\\\ -2 & 5 \\\\ 3 & -6 \\end{bmatrix}$

        6. $AI$

        &emsp; $\\begin{bmatrix} 1 & -2 & 3 \\\\ 4 & 5 & -6 \\end{bmatrix}$

        """)
    
    st.header("Part 2")

    st.markdown("""
        ## NBA Team Selector - Artificial Neural Network (ANN)
        
        This interactive tool uses a deep neural network to identify the 5 optimal NBA players from a pool of 100 based on their performance statistics.
        
        ### How the Model Works:
        
        **1. Architecture:**
        The ANN consists of 4 fully connected layers:
        - **Input Layer:** Accepts 1,200 features (12 statistics × 100 players)
        - **Hidden Layer 1:** 800 neurons with ReLU activation and 30% Dropout
        - **Hidden Layer 2:** 400 neurons with ReLU activation and 30% Dropout  
        - **Hidden Layer 3:** 100 neurons with ReLU activation and 30% Dropout
        - **Output Layer:** 100 neurons with Sigmoid activation (one per player)
        
        The Dropout layers prevent overfitting by randomly deactivating 30% of neurons during training.
        
        **2. Input Data:**
        For each of the 100 players, we use these 12 performance metrics:
        - Physical: Height, Weight
        - Games Played (GP)
        - Scoring & Rebounding: Points (PTS), Rebounds (REB), Assists (AST)
        - Advanced Stats: Net Rating, Offensive/Defensive Rebound %, Usage %, True Shooting %, Assist %
        
        **3. Training Process:**
        - We select 5 "optimal" players you choose and label them as 1.0
        - The remaining 95 players are labeled as 0.0
        - The model uses **Binary Cross-Entropy (BCE) Loss** to learn the distinction
        - **Adam Optimizer** adjusts weights and biases with a learning rate of 0.01
        - Training runs for a specified number of epochs (iterations)
        
        **4. Prediction:**
        - The model outputs a probability score (0-1) for each player
        - We extract the **Top 5 highest probability scores**
        - These 5 players are selected as the AI's optimal team
        
        **5. Loss Function:**
        $$\\text{BCE} = -\\frac{1}{N} \\sum_{i=1}^{N} [y_i \\log(\\hat{y}_i) + (1-y_i) \\log(1-\\hat{y}_i)]$$
        
        Where $y_i$ is the true label (1 or 0) and $\\hat{y}_i$ is the predicted probability.
        
        ### How to Use:
        1. Select up to 5 NBA players you consider optimal from the sidebar
        2. Adjust the number of samples and training epochs
        3. Click "Run Model" to train the network
        4. The model will display which players it selected as the optimal team
        
        ### Ethical Considerations:
        This model is trained on your personal selections, which means it reflects your biases and preferences. 
        The statistics-based approach cannot capture intangible qualities like leadership or chemistry, and should never be used for real hiring or trading decisions in professional sports, as such applications could perpetuate discrimination and unfairly impact players' careers and livelihoods. 
        Always treat ML predictions as a tool for exploration, not as ground truth for consequential decisions.
        """)
    

    # =====================================
    # AI Model, Data collection, Training, and Evaluation
    # =====================================
    nbaPlayerSet = pd.read_csv("./data/all_seasons.csv")

    # ==========================================
    # SIDEBAR: DATA SELECTOR AND OPTIMAL PLAYER SELECTOR
    # ==========================================

    # Allow users to configure parameters for synthetic data generation
    st.sidebar.header("NBA Player  Data Selector")

    # Slider to control the number of data points to use
    n_samples = st.sidebar.slider(
        "Number of Samples",
        min_value=50,
        max_value=200,
        value=100,
        step=10,
        help="Number of data points to use"
    )

    # Multiselect to choose optimal players
    # Create display labels with player name and season
    nbaPlayerSet['display_label'] = nbaPlayerSet['player_name'] + " (" + nbaPlayerSet['season'] + ")"
    
    selected_players = st.sidebar.multiselect(
        "Select Optimal Players (up to 5)",
        options=nbaPlayerSet['display_label'].unique(),
        max_selections=5,
        help="Select up to 5 players to represent the optimal team"
    )
    
    # Get the IDs of selected players
    selected_ids = []
    if selected_players and len(selected_players) == 5:
        selected_ids = nbaPlayerSet[nbaPlayerSet['display_label'].isin(selected_players)]['id'].unique().tolist()
        st.sidebar.info(f"Selected Player IDs: {selected_ids}")

    # Slider to control the number of training iterations (epochs) for ANN
    n_iterations = st.sidebar.slider(
        "Number of Epochs",
        min_value=10,
        max_value=1000,
        value=100,
        step=100,
        help="Number of loops to train the model"
    )

    # ===========================
    # DATA GENERATION
    # ===========================
    if st.sidebar.button("🎲 Run Model", type="primary"):
        # Create the pool of 100 players ensure our 5 optimal are included
        optimal_players = nbaPlayerSet[nbaPlayerSet['id'].isin(selected_ids)]
        other_players = nbaPlayerSet[~nbaPlayerSet['id'].isin(selected_ids)].sample(n=95, random_state=42)
        
        # combine and shuffle so model doesnt learn placement
        pool_100 = pd.concat([optimal_players, other_players]).sample(frac=1).reset_index(drop=True)

        # create mapping to make validation easier
        index_to_name = pool_100['player_name'].to_dict()
        index_to_id = pool_100['id'].to_dict()

        # Map our optimal_players to a target column
        pool_100['target'] = pool_100['id'].isin(selected_ids).astype(float)

        feature_cols = ["player_height", "player_weight", "gp", "pts", "reb", "ast", 
                    "net_rating", "oreb_pct", "dreb_pct", "usg_pct", "ts_pct", "ast_pct"]
         # Flatten X and y
        st.session_state.X = pool_100[feature_cols].values.flatten().reshape(1, -1)
        y = pool_100["target"].values.reshape(1, -1)

        st.session_state.model = NbaTeamSelectorANN(input_dim=len(feature_cols)*n_samples, num_classes=n_samples)
        st.session_state.losses = st.session_state.model.train_model(st.session_state.X, y, n_iterations)

        st.session_state.model_trained = True

        # Display success message to the user
        st.sidebar.success("✅ Data Trained!")

    if "model_trained" not in st.session_state:
        st.session_state.model_trained = False
        st.info("👆 Click 'Run Model' to generate Model data and results")
        st.stop()
    
    if st.session_state.model_trained == True and "model" in st.session_state:
        # Get the one-hot vector and the indices of the top 5
        one_hot, top_indices = st.session_state.model.predict(st.session_state.X)
        
        # Flatten top_indices to a simple list
        selected_indices = top_indices.flatten()
        
        # Map back to the dataframe
        picked_ids = [index_to_id[i] for i in selected_indices]
        picked_names = [index_to_name[i] for i in selected_indices]

        st.subheader("Model Results")
        
        # Display training loss graph
        if "losses" in st.session_state:
            loss_df = pd.DataFrame({
                "Epoch": range(1, len(st.session_state.losses) + 1),
                "Loss": st.session_state.losses
            })
            st.line_chart(
                loss_df.set_index("Epoch"),
                x_label= "Epoch",
                y_label="Loss"
            )

        st.subheader("The ML Starting Five")
        for name, pid in zip(picked_names, picked_ids):
            st.write(f"Player: {name} (ID: {pid})")

        # Validation logic using player names
        # We check if the name of the picked player is in our original selected players list
        selected_player_names = [name.split(" (")[0] for name in selected_players]  # Extract player names from display labels
        correct_picks = [name for name in picked_names if name in selected_player_names]
        st.write(f"\nAI correctly identified {len(correct_picks)}/5 designated optimal players.")
        if correct_picks:
            st.write(f"Correct Players: {correct_picks}")
    
    
    
Main()
