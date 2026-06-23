import io
import time
import xml.etree.ElementTree as ET
from collections import defaultdict

import networkx as nx
import pandas as pd
import requests


class KEGGNavigator:
    """
    A utility class to navigate KEGG pathways, parse KGML, and build gene interaction graphs.
    Supports both metabolic (reaction-based) and signaling (relation-based) pathways.
    """

    def __init__(self):
        self.base_url = "http://rest.kegg.jp"
        self.last_request_time = 0
        self.min_interval = 0.4  # Minimum 0.4s between requests (~2.5 req/s)

    def _wait_for_rate_limit(self):
        """Enforces rate limiting."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_request_time = time.time()

    def list_all_pathways_df(self, organism_code):
        """
        Lists all pathways for a given organism code (e.g., 'eco') and returns a DataFrame.
        Columns: 'pathway_id', 'description'
        """
        self._wait_for_rate_limit()
        url = f"{self.base_url}/list/pathway/{organism_code}"
        response = requests.get(url)

        if response.ok:
            df = pd.read_csv(
                io.StringIO(response.text),
                sep="\t",
                header=None,
                names=["pathway_id", "description"],
            )
            return df
        else:
            print(f"Error fetching pathway list: {response.status_code}")
            return pd.DataFrame()

    def get_pathway_kgml(self, pathway_id):
        """
        Query a specific pathway to get its KGML (XML) structure.
        """
        self._wait_for_rate_limit()
        url = f"{self.base_url}/get/{pathway_id}/kgml"
        response = requests.get(url)
        if response.ok:
            return response.text
        else:
            print(f"Error fetching KGML for {pathway_id}: {response.status_code}")
            return None

    def parse_kgml_entries(self, kgml_text):
        """
        Parses KGML to extract entries, reactions, and relations.
        Auto-detects whether the pathway is primarily metabolic or signaling based on content.

        Returns:
            dict with keys: 'entries', 'reactions', 'relations', 'type'
        """
        if not kgml_text:
            return {
                "entries": pd.DataFrame(),
                "reactions": pd.DataFrame(),
                "relations": pd.DataFrame(),
                "type": "unknown",
            }

        try:
            root = ET.fromstring(kgml_text)
        except ET.ParseError as e:
            print(f"XML Parse Error: {e}")
            return {
                "entries": pd.DataFrame(),
                "reactions": pd.DataFrame(),
                "relations": pd.DataFrame(),
                "type": "error",
            }

        # 1. Parse Entries
        entries = []
        for entry in root.findall("entry"):
            e_id = entry.get("id")
            e_name = entry.get("name")  # e.g. "eco:b1234"
            e_type = entry.get("type")
            e_reaction = entry.get("reaction")
            e_link = entry.get("link")

            graphics = entry.find("graphics")
            e_label = graphics.get("name") if graphics is not None else None

            entries.append(
                {
                    "id": e_id,
                    "name": e_name,
                    "type": e_type,
                    "reaction": e_reaction,
                    "label": e_label,
                    "link": e_link,
                }
            )
        df_entries = pd.DataFrame(entries)

        # 2. Parse Reactions (Metabolic)
        reactions = []
        for rn in root.findall("reaction"):
            rn_id = rn.get("name")
            rn_type = rn.get("type")
            substrates = [sub.get("name") for sub in rn.findall("substrate")]
            products = [prod.get("name") for prod in rn.findall("product")]

            reactions.append(
                {
                    "reaction_id": rn_id,
                    "type": rn_type,
                    "substrates": substrates,
                    "products": products,
                }
            )
        df_reactions = pd.DataFrame(reactions)

        # 3. Parse Relations (Signaling/Cellular)
        relations = []
        for rel in root.findall("relation"):
            entry1 = rel.get("entry1")
            entry2 = rel.get("entry2")
            rel_type = rel.get("type")

            # Subtypes (e.g., compound, activation, phosphorylation)
            subtypes = []
            for sub in rel.findall("subtype"):
                sub_name = sub.get("name")
                sub_value = sub.get("value")
                subtypes.append((sub_name, sub_value))

            relations.append(
                {"entry1": entry1, "entry2": entry2, "type": rel_type, "subtypes": subtypes}
            )
        df_relations = pd.DataFrame(relations)

        # 4. Auto-detect Type
        # Prioritize metabolic if reactions exist, otherwise signaling if relations exist
        if not df_reactions.empty:
            pathway_type = "metabolic"
        elif not df_relations.empty:
            pathway_type = "signaling"
        else:
            pathway_type = "other"

        return {
            "entries": df_entries,
            "reactions": df_reactions,
            "relations": df_relations,
            "type": pathway_type,
        }

    def build_gene_graph(self, parsed_data):
        """
        Constructs a gene interaction graph based on the parsed data type.
        Nodes are gene LABELS. Isolated nodes are removed.
        """
        p_type = parsed_data["type"]
        entries = parsed_data["entries"]

        if entries.empty:
            return nx.DiGraph()

        # Helper to map internal ID to Label
        id_to_label = {}
        gene_entries = entries[entries["type"] == "gene"]

        for _, row in gene_entries.iterrows():
            entry_id = row["id"]
            raw_label = row["label"]

            # Clean label: "fabG, ..." -> "fabG"
            if raw_label:
                label = raw_label.split(",")[0].strip()
                if label.endswith("..."):
                    label = label[:-3]
            else:
                # Fallback to name (e.g. eco:b1234) if no label
                label = row["name"].split()[0]

            id_to_label[entry_id] = label

        if p_type == "metabolic":
            return self._build_metabolic_graph(entries, parsed_data["reactions"], id_to_label)
        elif p_type == "signaling":
            return self._build_signaling_graph(entries, parsed_data["relations"], id_to_label)
        else:
            G = nx.DiGraph()
            return G

    def _build_metabolic_graph(self, df_entries, df_reactions, id_to_label):
        """
        Builds graph for metabolic pathways: Gene A -> Gene B via Compound.
        """
        if df_reactions.empty:
            return nx.DiGraph()

        # 1. Map Reaction IDs to Substrates and Products
        reaction_to_products = defaultdict(set)
        reaction_to_substrates = defaultdict(set)

        for _, row in df_reactions.iterrows():
            rn_ids = str(row["reaction_id"]).split()
            products = set(row["products"])
            substrates = set(row["substrates"])

            for rn_id in rn_ids:
                reaction_to_products[rn_id].update(products)
                reaction_to_substrates[rn_id].update(substrates)

        # 2. Map Genes to Produced and Consumed Compounds via Reactions
        # Note: df_entries has 'reaction' attribute linking gene to reaction ID
        gene_to_products = defaultdict(set)
        gene_to_substrates = defaultdict(set)

        # We need to map internal Entry ID -> Reaction -> Compound
        # The 'entries' df has 'id' (internal integer string) and 'reaction'

        for _, row in df_entries[df_entries["type"] == "gene"].iterrows():
            entry_id = row["id"]
            rn_str = str(row["reaction"])
            if rn_str == "nan" or not rn_str:
                continue

            rn_ids = rn_str.split()
            for rn_id in rn_ids:
                if rn_id in reaction_to_products:
                    gene_to_products[entry_id].update(reaction_to_products[rn_id])
                if rn_id in reaction_to_substrates:
                    gene_to_substrates[entry_id].update(reaction_to_substrates[rn_id])

        # 3. Build Graph
        G = nx.DiGraph()
        # Add all labels as nodes
        for label in set(id_to_label.values()):
            G.add_node(label)

        # Invert: Compound -> [Producer IDs]
        compound_to_producers = defaultdict(list)
        for g_id, prods in gene_to_products.items():
            for p in prods:
                compound_to_producers[p].append(g_id)

        # Edges: Consumer -> Substrate -> Producer
        for consumer_id, consumed_compounds in gene_to_substrates.items():
            consumer_label = id_to_label.get(consumer_id)
            if not consumer_label:
                continue

            for compound in consumed_compounds:
                producer_ids = compound_to_producers.get(compound, [])
                for producer_id in producer_ids:
                    if producer_id == consumer_id:
                        continue  # Skip self-loop

                    producer_label = id_to_label.get(producer_id)
                    if not producer_label:
                        continue

                    if producer_label == consumer_label:
                        continue  # Skip same-label loop

                    # Add/Update Edge
                    if G.has_edge(producer_label, consumer_label):
                        G[producer_label][consumer_label]["compounds"].add(compound)
                        # Update display string
                        G[producer_label][consumer_label]["compound"] = ", ".join(
                            sorted(G[producer_label][consumer_label]["compounds"])
                        )
                    else:
                        G.add_edge(
                            producer_label,
                            consumer_label,
                            compounds={compound},
                            compound=compound,
                            interaction="metabolic",
                        )

        # Remove isolated nodes
        G.remove_nodes_from(list(nx.isolates(G)))

        # 4. Annotate interactions with compound names
        all_compounds = set()
        for u, v, d in G.edges(data=True):
            if "compounds" in d:
                all_compounds.update(d["compounds"])

        # Batch fetch names
        if all_compounds:
            names_map = self.get_compound_names(list(all_compounds))

            for u, v, d in G.edges(data=True):
                if "compounds" in d:
                    # Get names for all compounds in this edge
                    c_ids = sorted(list(d["compounds"]))
                    c_names = [names_map.get(cid, cid) for cid in c_ids]
                    d["interaction_annotation"] = "; ".join(c_names)
                else:
                    d["interaction_annotation"] = ""

        return G

    def _build_signaling_graph(self, df_entries, df_relations, id_to_label):
        """
        Builds graph for signaling pathways: Gene A -> Gene B via Relation.
        """
        if df_relations.empty:
            return nx.DiGraph()

        G = nx.DiGraph()
        # Add nodes
        for label in set(id_to_label.values()):
            G.add_node(label)

        # Process Relations
        for _, rel in df_relations.iterrows():
            entry1 = rel["entry1"]
            entry2 = rel["entry2"]
            rel_type = rel["type"]
            subtypes = rel["subtypes"]  # list of (name, value)

            # Map IDs to Labels
            # Note: Relations might point to 'group' or 'map', handling simplified here to genes
            label1 = id_to_label.get(entry1)
            label2 = id_to_label.get(entry2)

            if label1 and label2 and label1 != label2:
                # Format subtype string
                subtype_str = ", ".join([f"{n}" for n, v in subtypes])

                if G.has_edge(label1, label2):
                    G[label1][label2]["types"].add(rel_type)
                    G[label1][label2]["subtypes"].add(subtype_str)
                else:
                    G.add_edge(
                        label1,
                        label2,
                        types={rel_type},
                        subtypes={subtype_str},
                        interaction="signaling",
                    )

        G.remove_nodes_from(list(nx.isolates(G)))

        # Annotate signaling interactions
        for u, v, d in G.edges(data=True):
            subtypes = list(d.get("subtypes", []))
            valid_subtypes = [s for s in subtypes if s]
            if valid_subtypes:
                d["interaction_annotation"] = ", ".join(valid_subtypes)
            else:
                types = list(d.get("types", []))
                d["interaction_annotation"] = ", ".join(types)

        return G

    def get_compound_names(self, compound_ids):
        """
        Fetches human-readable names for a list of compound IDs (e.g. ['cpd:C00001']).
        Returns a dict: {compound_id: name_string}
        """
        results = {}
        unique_ids = sorted(list(set([cid for cid in compound_ids if cid])))

        if not unique_ids:
            return results

        # Batch requests (max 10 for KEGG REST get)
        batch_size = 10
        for i in range(0, len(unique_ids), batch_size):
            batch = unique_ids[i : i + batch_size]
            query_ids = "+".join(batch)
            self._wait_for_rate_limit()
            url = f"{self.base_url}/get/{query_ids}"

            try:
                response = requests.get(url)
                if not response.ok:
                    print(f"Error fetching compounds {batch}: {response.status_code}")
                    continue

                current_entry_id = None
                current_names = []
                in_names = False

                for line in response.text.splitlines():
                    if line.startswith("///"):
                        # End of entry
                        if current_entry_id:
                            full_name = "; ".join(current_names)
                            # Assign to all matching input IDs
                            for q_id in batch:
                                # Check if q_id matches current_entry_id (with or without prefix)
                                # e.g. q_id="cpd:C00001", entry="C00001"
                                if q_id == current_entry_id or q_id.endswith(
                                    ":" + current_entry_id
                                ):
                                    results[q_id] = full_name

                        current_entry_id = None
                        current_names = []
                        in_names = False
                        continue

                    if line.startswith("ENTRY"):
                        parts = line.split()
                        if len(parts) >= 2:
                            current_entry_id = parts[1]
                        in_names = False
                    elif line.startswith("NAME"):
                        # "NAME        Name1;"
                        parts = line.split(maxsplit=1)
                        if len(parts) > 1:
                            val = parts[1].strip()
                            if val.endswith(";"):
                                val = val[:-1]
                            current_names.append(val)
                            in_names = True
                    elif in_names and line.startswith(" "):
                        val = line.strip()
                        if val.endswith(";"):
                            val = val[:-1]
                        current_names.append(val)
                    elif line and not line.startswith(" "):
                        in_names = False

            except Exception as e:
                print(f"Exception fetching compounds: {e}")

        return results


if __name__ == "__main__":
    navigator = KEGGNavigator()

    print("--- 1. Fetching Pathway List ---")
    df = navigator.list_all_pathways_df("eco")
    print(f"Found {len(df)} pathways.")
    print(df.head())

    # 2. Metabolic Example (Glycolysis)
    metabolic_id = "eco00010"
    print(f"\n--- 2. Metabolic Example: {metabolic_id} ---")
    kgml_met = navigator.get_pathway_kgml(metabolic_id)
    parsed_met = navigator.parse_kgml_entries(kgml_met)
    print(f"Detected Type: {parsed_met['type']}")
    G_met = navigator.build_gene_graph(parsed_met)
    print(f"Graph: {G_met.number_of_nodes()} nodes, {G_met.number_of_edges()} edges")
    if G_met.number_of_edges() > 0:
        print("Sample edges:")
        for u, v, d in list(G_met.edges(data=True))[:3]:
            print(f"  {u} -> {v} ({d.get('compound')})")

    # 3. Signaling Example (Two-component system)
    signaling_id = "eco02020"
    print(f"\n--- 3. Signaling Example: {signaling_id} ---")
    kgml_sig = navigator.get_pathway_kgml(signaling_id)
    parsed_sig = navigator.parse_kgml_entries(kgml_sig)
    print(f"Detected Type: {parsed_sig['type']}")
    G_sig = navigator.build_gene_graph(parsed_sig)
    print(f"Graph: {G_sig.number_of_nodes()} nodes, {G_sig.number_of_edges()} edges")
    if G_sig.number_of_edges() > 0:
        print("Sample edges:")
        for u, v, d in list(G_sig.edges(data=True))[:3]:
            # Convert sets to string for printing
            types_str = list(d.get("types"))[0]
            print(f"  {u} -> {v} ({types_str})")
