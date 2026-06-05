import torch
from evo2 import Evo2

LAYER = "blocks.28.mlp.l3"

model = Evo2("evo2_7b")
model.model.eval()
DEVICE = next(model.model.parameters()).device  # follows the model, not hardcoded


def embed_gene(
    seq: str,
    layer: str = LAYER,
    pool: str = "mean",
    both_strands: bool = True,
) -> torch.Tensor:
    valid = set("ACGTacgt")
    bad = set(seq) - valid
    if bad:
        raise ValueError(f"Sequence contains non-ACGT characters: {bad!r}")

    def _forward(s: str) -> torch.Tensor:
        ids = torch.tensor(model.tokenizer.tokenize(s), dtype=torch.int).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            _, emb = model(ids, return_embeddings=True, layer_names=[layer])
        return emb[layer].squeeze(0).float().cpu()

    h = _forward(seq)

    if both_strands:
        rc = seq.translate(str.maketrans("ACGTacgt", "TGCAtgca"))[::-1]
        h_rc = _forward(rc)
        h = (h + h_rc.flip(0)) / 2

    if pool == "mean":
        return h.mean(dim=0)
    elif pool == "last":
        return h[-1]
    elif pool == "none":
        return h
    else:
        raise ValueError(f"Unknown pool mode: {pool!r}")


gene_seq = (
    "ATGAAACCAGTAACGTTATACGATGTCGCAGAGTATGCCGGTGTCTCTTATCAGACCGTTTCCCGCGTGGTG"
    "AACCAGGCCAGCCACGTTTCTGCGAAAACGCGGGAAAAAGTGGAAGCGGCGATGGCGGAGCTGAATTACATT"
    "CCCAACCGCGTGGCACAACAACTGGCGGGCAAACAGTCGTTGCTGATTGGCGTTGCCACCTCCAGTCTGGCC"
    "CTGCACGCGCCGTCGCAAATTGTCGCGGCGATTAAATCTCGCGCCGATCAACTGGGTGCCAGCGTGGTGGTG"
)

print(f"Model loaded on: {DEVICE}")
print("Running embed_gene smoke test...")

vec_mean = embed_gene(gene_seq, pool="mean")
assert vec_mean.ndim == 1, f"Expected 1D, got {vec_mean.shape}"

vec_none = embed_gene(gene_seq, pool="none")
assert vec_none.ndim == 2, f"Expected 2D, got {vec_none.shape}"
assert vec_none.shape[0] == len(
    gene_seq
), f"Token count {vec_none.shape[0]} != seq len {len(gene_seq)}"

print(f"mean-pooled embedding : {vec_mean.shape}")
print(f"per-token embeddings  : {vec_none.shape}")
print(f"embedding norm        : {vec_mean.norm():.4f}")
print("PASS")
