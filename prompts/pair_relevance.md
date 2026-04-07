**System:**

You are a microbial metabolism expert.

**User:**

I am studying functional coupling between metabolic genes in *E. coli* K-12. Two genes are considered "functionally coupled" if knocking down one would be expected to directly affect the metabolic function of the other, e.g., because they catalyze consecutive or tightly connected reactions in a pathway that is active during growth in rich medium (LB).

Below is a list of gene pairs. For each pair, decide whether the two genes are functionally coupled under these conditions. Return 1 if yes, 0 if no.

Be stringent: two genes belonging to the same broad functional category (e.g., both involved in amino acid metabolism) is not sufficient. There should be a concrete mechanistic reason why perturbing one would directly impact the other's metabolic role. Also consider whether the relevant reactions are actually active in rich medium — pairs involving biosynthetic pathways whose products are abundantly supplied in LB should be scored 0.

Gene pairs:
{pairs_list}

Return ONLY a JSON object mapping each pair to its score, no commentary:

```json
{"geneA_geneB": 0, "geneC_geneD": 1, ...}
```
