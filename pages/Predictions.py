import streamlit as st
import pandas as pd
import pickle
import numpy as np
import base64

def set_background(image_file):
    with open(image_file, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;s
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


set_background(r"C:\Users\Dell\Downloads\Download premium vector of Hand drawn travel element background vector set by marinemynt about background, cartoon, paper, compass, and cute 936632.jpg")
# ------------------------------
# Load models and encoders
# ------------------------------
@st.cache_data
def load_assets():
    with open("Ratings_model.pkl", "rb") as f: ratings_model = pickle.load(f)
    with open("VisitMode_model.pkl", "rb") as f: visit_mode_model = pickle.load(f)
    with open("feature_ordinal_encoder_cls.pkl", "rb") as f: feature_encoder = pickle.load(f)
    with open("target_label_encoder_cls.pkl", "rb") as f: target_encoder = pickle.load(f)
    with open("content_recommendation_model.pkl", "rb") as f: content_model = pickle.load(f)
    with open("collaborative_recommendation.pkl", "rb") as f: collab_model = pickle.load(f)
    return ratings_model, visit_mode_model, feature_encoder, target_encoder, content_model, collab_model

ratings_model, visit_mode_model, feature_encoder, target_encoder, content_model, collab_model = load_assets()
tour_df = content_model['tour_df']

# ------------------------------
# Layout and Tabs
# ------------------------------
st.title("🔍 Tourism Experience Predictions")

tab1, tab2, tab3 = st.tabs(["⭐ Rating Prediction", "👥 Visit Mode Classification", "🎥 Attraction Recommendation"])

# ------------------------------
# ⭐ TAB 1: Rating Prediction
# ------------------------------
with tab1:
    st.subheader("⭐ Predict Tourist Attraction Rating")

    # -- Visit year & month selection
    visit_year = st.number_input("Select Visit Year", min_value=2013, max_value=2022, step=1)
    visit_month = st.selectbox("Select Visit Month", list(range(1, 13)), format_func=lambda x: f"{x:02d}")

    # -- City and attraction selection
    cities = sorted(tour_df["CityName"].dropna().unique())
    city = st.selectbox("Select City", ["Select..."] + cities)

    if city != "Select...":
        city_df = tour_df[tour_df["CityName"] == city]
        attractions = sorted(city_df["Attraction"].dropna().unique())
        attraction = st.selectbox("Select Attraction", ["Select..."] + attractions)
    else:
        attraction = "Select..."

    # -- Visit mode selection
    visit_modes = sorted(tour_df["VisitMode"].dropna().unique())
    visit_mode = st.selectbox("Select Visit Mode", ["Select..."] + visit_modes)

    # -- Predict button
    if st.button("Predict Rating"):
        if city == "Select..." or attraction == "Select..." or visit_mode == "Select...":
            st.warning("Please select valid City, Attraction, and Visit Mode.")
        else:
            try:
                row = city_df[city_df["Attraction"] == attraction].iloc[0]
                input_df = pd.DataFrame([{
                    "VisitYear": int(visit_year),
                    "VisitMonth": int(visit_month),
                    "VisitMode": str(visit_mode),
                    "Continent": str(row["Continent"]),
                    "Region": str(row["Region"]),
                    "Country": str(row["Country"]),
                    "CityName": str(row["CityName"]),
                    "AttractionType": str(row["AttractionType"]),
                    "Attraction": str(attraction)
                }])[ratings_model.feature_names_]  # enforce column order

                pred = ratings_model.predict(input_df)[0]
                st.success(f"🌟 Predicted Rating: **{int(pred[0])}**")


            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")

           
# ------------------------------
# 👥 TAB 2: Visit Mode Classification
# ------------------------------
with tab2:
    st.subheader("👥 Predict Visit Mode")

    countries = sorted(tour_df["Country"].dropna().unique())
    country = st.selectbox("Select Country", ["Select..."] + countries)

    if country != "Select...":
        country_df = tour_df[tour_df["Country"] == country]
        user_counts = country_df["UserId"].value_counts().reset_index()
        user_counts.columns = ["UserId", "VisitCount"]
        visit_count_options = sorted(user_counts["VisitCount"].unique())
        selected_count = st.selectbox("Select User Visit Count", ["Select..."] + [str(v) for v in visit_count_options])
    else:
        selected_count = "Select..."

    if st.button("Predict Visit Mode"):
        if country == "Select..." or selected_count == "Select...":
            st.warning("Please select valid Country and Visit Count.")
        else:
            try:
                selected_count = int(selected_count)
                sample_user = user_counts[user_counts["VisitCount"] == selected_count]["UserId"].iloc[0]
                row = country_df[country_df["UserId"] == sample_user].iloc[0]

                continent = row["Continent"]
                popularity = country_df["Attraction"].value_counts().mean()

                # --- Construct input DataFrame
                input_df = pd.DataFrame([{
                    "Continent": continent,
                    "Country": country,
                    "User_Visit_Count": selected_count,
                    "Attraction_Popularity": popularity
                }])

                # --- Apply ordinal encoding to categorical columns
                cat_cols = ["Continent", "Country"]
                encoded_cats = feature_encoder.transform(input_df[cat_cols])

                # --- Combine with numeric columns
                numeric_cols = input_df[["User_Visit_Count", "Attraction_Popularity"]].values
                final_input = np.hstack([encoded_cats, numeric_cols])

                # --- Predict
                pred_encoded = visit_mode_model.predict(final_input)
                pred_label = target_encoder.inverse_transform(pred_encoded)[0]

                st.success(f"🧭 Predicted Visit Mode: **{pred_label}**")

            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")

# ------------------------------
# 🎥 TAB 3: Recommendation
# ------------------------------
with tab3:
    st.subheader("🎥 Personalized Recommendations")

    user_ids = sorted(tour_df["UserId"].unique())
    user_id = st.selectbox("Select User ID", ["Select..."] + list(map(int, user_ids)))

    if user_id == "Select...":
        st.warning("Please select a valid User ID.")
    else:
        if st.button("Generate Recommendations"):

            # --- Content-Based ---
            def get_content_recommendations(user_id, top_n=5):
                tfidf_vectorizer = content_model['tfidf_vectorizer']
                nn_model = content_model['nn_model']
                attraction_to_index = content_model['attraction_to_index']
                visited = tour_df[tour_df['UserId'] == user_id]['Attraction'].unique()

                recs = []
                for attraction in visited:
                    if attraction not in attraction_to_index:
                        continue
                    idx = attraction_to_index[attraction]
                    distances, indices = nn_model.kneighbors(
                        tfidf_vectorizer.transform([tour_df.loc[idx, 'Content_Features']]),
                        n_neighbors=top_n + 1
                    )
                    for i, (dist, neighbor_idx) in enumerate(zip(distances[0], indices[0])):
                        if i == 0 or tour_df.iloc[neighbor_idx]['Attraction'] in visited:
                            continue
                        sim = 1 - dist
                        recs.append({
                            "Attraction": tour_df.iloc[neighbor_idx]['Attraction'],
                            "Similarity": sim,
                            "Reason": f"Similar to {attraction}"
                        })

                # Return safe empty DataFrame if no recommendations
                if not recs:
                    return pd.DataFrame(columns=["Attraction", "Similarity", "Reason"])

                df = pd.DataFrame(recs)
                return df.drop_duplicates("Attraction").sort_values(by="Similarity", ascending=False).head(top_n)

            # --- Collaborative-Based ---
            def get_collaborative_recommendations(user_id, n=5):
                user_item_matrix = collab_model['user_item_matrix']
                user_means = collab_model['user_means']
                sim_df = collab_model['user_similarity_df']

                if user_id not in user_item_matrix.index or user_id not in sim_df.index:
                    return []

                rated = user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index

                preds = []
                for att_id in user_item_matrix.columns:
                    if att_id in rated:
                        continue

                    sim_users = sim_df[user_id].sort_values(ascending=False)[1:6]
                    ratings = user_item_matrix.loc[sim_users.index, att_id]
                    sim_scores = sim_users[ratings > 0]
                    ratings = ratings[ratings > 0]

                    if len(ratings) == 0:
                        continue

                    pred = np.dot(sim_scores, ratings) / sim_scores.sum()
                    attraction_name = tour_df[tour_df['AttractionId'] == att_id]['Attraction'].values
                    if len(attraction_name) > 0:
                        preds.append((attraction_name[0], pred))

                return sorted(preds, key=lambda x: x[1], reverse=True)[:n]

            # --- Display Content-Based Recommendations ---
            st.markdown("#### 📌 Content-Based Recommendations:")
            cb_recs = get_content_recommendations(user_id)
            if cb_recs.empty:
                st.warning("No content-based recommendations found for this user.")
            else:
                for i, row in cb_recs.iterrows():
                    st.write(f"{i+1}. **{row['Attraction']}** – _{row['Reason']}_ (Similarity: {row['Similarity']:.2f})")

            # --- Display Collaborative Recommendations ---
            st.markdown("#### 👥 Collaborative Recommendations:")
            collab_recs = get_collaborative_recommendations(user_id)
            if not collab_recs:
                st.warning("No collaborative recommendations found for this user.")
            else:
                for i, (name, pred) in enumerate(collab_recs, 1):
                    st.write(f"{i}. **{name}** (Predicted Rating: {pred:.2f})")
