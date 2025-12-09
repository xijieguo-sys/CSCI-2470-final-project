import os
from typing import List, Optional

import numpy as np
import scanpy as sc
from tqdm import tqdm

import google.generativeai as genai



def load_and_preprocess_h5ad(h5ad_path: str) -> sc.AnnData:
    print(f"[INFO] Loading {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    print("[INFO] Preprocessing done.")
    return adata


def build_cell_sentence(expr_row, gene_names, max_genes=100):
    if hasattr(expr_row, "toarray"):
        expr_row = expr_row.toarray().ravel()

    nz = np.where(expr_row > 0)[0]
    if len(nz) == 0:
        return "This cell shows no detected gene expression."

    sorted_idx = nz[np.argsort(expr_row[nz])[::-1]]
    sorted_idx = sorted_idx[:max_genes]

    selected_genes = gene_names[sorted_idx]
    gene_list = ", ".join(selected_genes)

    return f"This single cell highly expresses the following genes: {gene_list}."


def build_sentences_for_all_cells(adata, max_genes=100):
    sentences = []
    gene_names = np.array(adata.var_names)

    print("[INFO] Building sentences...")
    for i in tqdm(range(adata.n_obs)):
        row = adata.X[i]
        sent = build_cell_sentence(row, gene_names, max_genes)
        sentences.append(sent)

    return sentences


def gemini_embed_sentences(sentences: List[str], batch_size=32):
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key is None:
        raise RuntimeError("GOOGLE_API_KEY is not set")

    genai.configure(api_key=api_key)

    embeddings = []
    print("[INFO] Embedding sentences using Gemini (models/embedding-001)...")

    for start in tqdm(range(0, len(sentences), batch_size)):
        batch = sentences[start:start + batch_size]

        response = genai.embed_content(
            model="models/gemini-embedding-001",
            content=batch
        )
        embeds = response["embedding"]
        embeddings.extend(embeds)

    return np.array(embeddings, dtype=np.float32)



def genept_s_pipeline_gemini(
    h5ad_in: str,
    h5ad_out: str,
    max_genes=100,
    batch_size=32
):
    adata = load_and_preprocess_h5ad(h5ad_in)

    sentences = build_sentences_for_all_cells(adata, max_genes=max_genes)

    embeddings = gemini_embed_sentences(sentences, batch_size=batch_size)

    adata.obsm["X_genept_s_gemini"] = embeddings

    print(f"[INFO] Saving updated file to {h5ad_out}")
    adata.write_h5ad(h5ad_out)

    return adata


if __name__ == "__main__":

    input_dir = "data"
    output_dir = "output_data/genept_s_embeddings"
    os.makedirs(output_dir, exist_ok=True)

    sample_files = sorted([
        f for f in os.listdir(input_dir)
        if f.endswith(".h5ad")
    ])

    print("[INFO] Found samples:")
    for f in sample_files:
        print(" -", f)
    
        
    for f in sample_files:
        sample_name = f[:-5]
        h5ad_in = os.path.join(input_dir, f)
        h5ad_out = os.path.join(output_dir, f"{sample_name}_genept_s.h5ad")

        try:
            genept_s_pipeline_gemini(
                h5ad_in=h5ad_in,
                h5ad_out=h5ad_out,
                max_genes=100,
                batch_size=16
            )
        except Exception as e:
            print(f"[ERROR] Failed on {sample_name}: {e}")
            continue

    print("\n===============================")
    print("[INFO All samples processed!")
    print("===============================")

    # input_file = "data/DLPFC_151674.h5ad"
    # output_file = "DLPFC_151674_genept_s.h5ad"

    # adata_new = genept_s_pipeline_gemini(
    #     h5ad_in=input_file,
    #     h5ad_out=output_file,
    #     max_genes=100,
    #     batch_size=10
    # )

    # print("[INFO] DONE!")
