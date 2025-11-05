# app.py — Streamlit recommender (artifacts in same folder)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import gzip, shutil
from pathlib import Path

st.set_page_config(page_title="Product Recommender", layout="wide")

@st.cache_resource
def load_artifacts():
    root = Path(".")
    artifacts = {}

    # --- product_meta ---
    pm_path = root / "product_meta_small.joblib"
    pm_pickle = root / "product_meta_small.pkl"
    if pm_path.exists():
        try:
            pm = joblib.load(pm_path)
            # joblib may return a DataFrame or a dict-like
            if isinstance(pm, pd.DataFrame):
                artifacts["product_meta"] = pm
            else:
                # try convert to DataFrame
                try:
                    artifacts["product_meta"] = pd.DataFrame(pm)
                except Exception:
                    artifacts["product_meta"] = None
        except Exception:
            artifacts["product_meta"] = None
    elif pm_pickle.exists():
        try:
            artifacts["product_meta"] = pd.read_pickle(pm_pickle)
        except Exception:
            artifacts["product_meta"] = None
    else:
        artifacts["product_meta"] = None

    # --- similarity_map ---
    sm_path = root / "similarity_map_small.joblib"
    artifacts["similarity_map"] = joblib.load(sm_path) if sm_path.exists() else None

    # --- tfidf ---
    tf_path = root / "tfidf_vectorizer.joblib"
    artifacts["tfidf"] = joblib.load(tf_path) if tf_path.exists() else None

    # --- svd_model_small (to infer annoy dim) ---
    svd_path = root / "svd_model_small.joblib"
    svd_obj = None
    if svd_path.exists():
        try:
            svd_job = joblib.load(svd_path)
            # earlier code saved dict {'svd': svd_small} sometimes
            if isinstance(svd_job, dict) and 'svd' in svd_job:
                svd_obj = svd_job['svd']
            else:
                svd_obj = svd_job
        except Exception:
            svd_obj = None
    artifacts["svd_model"] = svd_obj

    # --- Annoy: decompress if needed then load path (we'll lazily load Annoy index) ---
    ann_gz = root / "annoy_index_small.ann.gz"
    ann_file = root / "annoy_index_small.ann"
    if ann_gz.exists() and not ann_file.exists():
        try:
            with gzip.open(ann_gz, "rb") as f_in, open(ann_file, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            artifacts["annoy_decompressed"] = True
        except Exception:
            artifacts["annoy_decompressed"] = False
    artifacts["annoy_path"] = str(ann_file) if ann_file.exists() else None

    return artifacts

art = load_artifacts()
product_meta = art.get("product_meta")
similarity_map = art.get("similarity_map")
tfidf_vec = art.get("tfidf")
svd_model = art.get("svd_model")
annoy_path = art.get("annoy_path")

# Annoy will be loaded lazily only if needed
_annoy_index = None
def load_annoy_index():
    global _annoy_index
    if _annoy_index is not None:
        return _annoy_index
    if annoy_path is None:
        return None
    try:
        from annoy import AnnoyIndex
    except Exception:
        return None
    # infer dim from svd_model if available
    dim = 32
    if svd_model is not None and hasattr(svd_model, "n_components"):
        dim = int(svd_model.n_components)
    ai = AnnoyIndex(dim, metric="angular")
    ai.load(annoy_path)
    _annoy_index = ai
    return _annoy_index

# --- Basic checks ---
st.title("🛍️ Product Recommender")
st.markdown("Lightweight content-based recommendations. Uses `similarity_map_small.joblib` if present, otherwise Annoy index.")

if product_meta is None:
    st.error("product_meta_small.joblib (or product_meta_small.pkl) not found in repository root. Please add it.")
    st.stop()

# Ensure product_meta columns exist
if 'product_id' not in product_meta.columns:
    product_meta['product_id'] = product_meta.index.astype(str)
if 'title' not in product_meta.columns:
    product_meta['title'] = product_meta['product_id'].astype(str)

product_meta['product_id'] = product_meta['product_id'].astype(str)
product_ids = product_meta['product_id'].tolist()
prod_to_idx = {p: i for i,p in enumerate(product_ids)}

def get_title(pid):
    row = product_meta.loc[product_meta['product_id'] == str(pid)]
    return row['title'].iloc[0] if not row.empty else str(pid)

# Try to detect an image column for nicer UI
image_col = None
for c in ['image','img','thumbnail','url','link']:
    if c in product_meta.columns:
        image_col = c
        break

# --- recommendation helpers ---
def similar_by_map(pid, k=5):
    pid = str(pid)
    if similarity_map is None:
        return None
    return similarity_map.get(pid, [])[:k]

def similar_by_annoy(pid, k=5):
    ai = load_annoy_index()
    if ai is None:
        return None
    idx = prod_to_idx.get(str(pid))
    if idx is None:
        return []
    nbrs = ai.get_nns_by_item(idx, k+1, include_distances=False)
    # exclude self
    nbrs = [product_ids[i] for i in nbrs if product_ids[i] != pid][:k]
    return nbrs

def get_similar(pid, k=5):
    # prefer similarity_map (small & exact), else use annoy
    res = similar_by_map(pid, k)
    if res is not None:
        return res
    return similar_by_annoy(pid, k) or []

# --- Sidebar UI ---
with st.sidebar:
    st.header("Query")
    query = st.text_input("Search product title", "")
    top_k = st.slider("Number of similar products", 1, 20, 6)
    show_meta = st.checkbox("Show metadata (IDs, etc.)", value=False)

# --- center UI: product search / selection ---
st.subheader("Select a product to get similar items")

selected_pid = None
if query.strip():
    mask = product_meta['title'].str.contains(query, case=False, na=False)
    matches = product_meta[mask].head(100)
    if matches.empty:
        st.warning("No matching products found for that query.")
    else:
        opts = matches['product_id'].tolist()
        selected_pid = st.selectbox("Choose product", options=opts, format_func=get_title)
else:
    # show a few popular / top rows
    st.info("No search query — choose from top products")
    default_options = product_meta.head(100)['product_id'].tolist()
    selected_pid = st.selectbox("Choose product", default_options, format_func=get_title)

if selected_pid:
    st.markdown("### Selected")
    sel_row = product_meta[product_meta['product_id'] == str(selected_pid)].iloc[0]
    st.markdown(f"**{get_title(selected_pid)}**")
    if image_col and pd.notna(sel_row.get(image_col)):
        try:
            st.image(sel_row.get(image_col), width=240)
        except Exception:
            pass
    if show_meta:
        st.write(sel_row.to_dict())

    st.markdown("---")
    st.subheader("Similar products")
    recs = get_similar(selected_pid, k=top_k)
    if not recs:
        st.info("No similar products found (similarity_map missing and annoy not available).")
    else:
        # display grid of results
        cols = st.columns(2 if top_k <= 4 else 3)
        for i, pid in enumerate(recs):
            c = cols[i % len(cols)]
            with c:
                st.markdown(f"**{i+1}. {get_title(pid)}**")
                if image_col:
                    row = product_meta.loc[product_meta['product_id'] == pid]
                    if not row.empty and pd.notna(row.iloc[0].get(image_col)):
                        try:
                            st.image(row.iloc[0].get(image_col), width=200)
                        except Exception:
                            pass
                if show_meta:
                    row = product_meta.loc[product_meta['product_id'] == pid]
                    if not row.empty:
                        st.write(row.iloc[0].to_dict())

st.markdown("---")
st.caption("Artifacts loaded from repository root. If you want collaborative/hybrid features, add a `svd_factors.joblib` or `interactions.csv`.")
