# recommendation_sys

### 🚀 Live Demo
👉 [Open Recommender App on Streamlit] https://lsvwykxlplc5jnjiqbik4w.streamlit.app/

How to Use

Search Products:
Use the sidebar search bar to enter keywords (e.g. “Headphones”, “Shoes”, “AC”).
The app will show matching product titles from the dataset.

Select a Product:
Choose a product from the dropdown list to see its details.

Get Recommendations:
The app will display similar products using either:
Precomputed similarity map (similarity_map_small.joblib)
Or Annoy index for fast nearest-neighbor search.

Tune Recommendations:
Use the slider to choose number of recommendations (1–20).
Toggle “Show Metadata” to display full product info (ID, category, etc.).

Example Searches
Try searching:
“Air Conditioner”
“Headphones”
“Shoes”
“Smartphone”
and view instant top-5 similar product recommendations.
