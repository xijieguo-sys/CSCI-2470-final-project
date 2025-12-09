import os
from typing import Dict, List

import numpy as np
import scanpy as sc
import pickle
from tqdm import tqdm
import scipy.sparse as sp
import google.generativeai as genai
import json



API_KEY = os.getenv("GOOGLE_API_KEY")

if API_KEY is None:
    raise RuntimeError("GOOGLE_API_KEY is not set. Please export it in your environment.")


def load_and_preprocess_h5ad(h5ad_path: str) -> sc.AnnData:
    print(f"[INFO] Loading {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    print("[INFO] Preprocessing done.")
    return adata

def load_gene_summaries(summary_json):
    print(f"[INFO] Loading gene summaries from {summary_json}")
    with open(summary_json, "r") as f:
        summary_dict = json.load(f)
    print(f"[INFO] Loaded {len(summary_dict)} gene summaries.")
    return summary_dict

def embed_gene_summaries_with_gemini(
    summary_dict,
    batch_size=32,
    model="models/gemini-embedding-001"
):
    genai.configure(api_key=API_KEY)

    genes, summaries = [], []
    for gene, summary in summary_dict.items():
        if summary.strip():
            genes.append(gene)
            summaries.append(summary)

    print(f"[INFO] Embedding {len(genes)} gene summaries...")

    gene_embeddings = {}
    for start in tqdm(range(0, len(genes), batch_size)):
        batch_genes = genes[start:start + batch_size]
        batch_summaries = summaries[start:start + batch_size]

        response = genai.embed_content(
            model=model,
            content=batch_summaries
        )
        batch_embs = response["embedding"]

        for g, emb in zip(batch_genes, batch_embs):
            gene_embeddings[g] = np.asarray(emb, dtype=np.float32)

    return gene_embeddings

def align_genes(adata, gene_emb):
    adata_genes = list(adata.var_names)
    common_genes = [g for g in adata_genes if g in gene_emb]

    print(f"[INFO] AnnData genes:     {len(adata_genes)}")
    print(f"[INFO] Embedded genes:    {len(gene_emb)}")
    print(f"[INFO] Common genes:      {len(common_genes)}")

    if len(common_genes) == 0:
        raise RuntimeError("No genes overlap between AnnData and embeddings.")

    adata_sub = adata[:, common_genes].copy()
    return adata_sub, common_genes

def compute_genept_w_embeddings(adata_sub, common_genes, gene_emb):
    X = adata_sub.X
    if sp.issparse(X):
        X = X.toarray()

    gene_emb_matrix = np.vstack([gene_emb[g] for g in common_genes])

    row_sums = X.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    weights = X / row_sums

    cell_emb = weights @ gene_emb_matrix

    norms = np.linalg.norm(cell_emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    cell_emb_norm = cell_emb / norms

    return cell_emb_norm

def genept_w_pipeline(
    h5ad_in: str,
    summary_json: str,
    h5ad_out: str,
    batch_size=32,
):
    adata = load_and_preprocess_h5ad(h5ad_in)

    summary_dict = load_gene_summaries(summary_json)

    gene_emb = embed_gene_summaries_with_gemini(summary_dict, batch_size=batch_size)

    adata_sub, common_genes = align_genes(adata, gene_emb)

    print("[INFO] Computing GenePT-w cell embeddings...")
    cell_emb = compute_genept_w_embeddings(adata_sub, common_genes, gene_emb)

    adata.obsm["X_genept_w"] = cell_emb

    print(f"[INFO] Final shape: {cell_emb.shape}")
    print(f"[INFO] Saving to {h5ad_out}")
    adata.write_h5ad(h5ad_out)

    return adata

if __name__ == "__main__":
    input_file   = "data/DLPFC_151507.h5ad"
    summary_json = "output_data/ncbi_gene_summaries_latest.json"
    output_file  = "output_genept_w.h5ad"

    genept_w_pipeline(
        h5ad_in=input_file,
        summary_json=summary_json,
        h5ad_out=output_file,
        batch_size=16
    )

