import os
import math
import random

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import hashlib

from torch_scatter import scatter_add
_DEBUG_BATCH_ID = 0
_BATCH_COUNTER = 0
_EVAL_BATCH_COUNTER = 0
BASE_SEED = 12345 #(main: 12345)


def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)

def edge_normalization1(edge_type, edge_index, num_entity, num_relation):
    """
    Same semantics as your original:
      deg[src, rel] = count of edges with (src, rel)
      edge_norm[e] = 1 / deg[src_e, rel_e]

    But WITHOUT one_hot (saves huge time/memory).
    """
    device = edge_type.device
    src = edge_index[0].long()
    rel = edge_type.long()

    # linearize (src, rel) into one index
    idx = src * int(num_relation) + rel   # [E]
    ones = torch.ones(idx.numel(), device=device, dtype=torch.float)

    deg = scatter_add(ones, idx, dim=0, dim_size=int(num_entity) * int(num_relation))
    edge_norm = 1.0 / deg[idx].clamp_min_(1.0)   # avoid div by 0 (shouldn’t happen)
    return edge_norm

import torch
from torch_scatter import scatter_add

def edge_normalization(edge_type, edge_index, num_entity, num_relation):
    """
    R-GCN normalization (mean per relation for each target node):
      deg[tgt, rel] = #incoming edges into tgt with relation rel
      edge_norm[e]  = 1 / deg[tgt_e, rel_e]
    """
    tgt = edge_index[1].long()          # <-- IMPORTANT: target, not source
    rel = edge_type.long()

    idx = tgt * int(num_relation) + rel
    ones = torch.ones(idx.numel(), device=edge_type.device, dtype=torch.float)

    deg = scatter_add(
        ones, idx, dim=0,
        dim_size=int(num_entity) * int(num_relation)
    )

    edge_norm = 1.0 / deg[idx].clamp_min(1.0)
    return edge_norm



def sort_and_rank(score, target):
    _, indices = torch.sort(score, dim=1, descending=True)
    indices = torch.nonzero(indices == target.view(-1, 1))
    indices = indices[:, 1].view(-1)
    return indices


# return MRR (filtered), and Hits @ (1, 3, 10)
def calc_mrr(embedding, w, test_triplets, all_triplets, hits=[]):
    with torch.no_grad():

        num_entity = len(embedding)

        ranks_s = []
        ranks_o = []

        head_relation_triplets = all_triplets[:, :2]
        tail_relation_triplets = torch.stack((all_triplets[:, 2], all_triplets[:, 1])).transpose(0, 1)

        for test_triplet in tqdm(test_triplets):
            # Perturb object
            subject = test_triplet[0]
            relation = test_triplet[1]
            object_ = test_triplet[2]

            subject_relation = test_triplet[:2]
            delete_index = torch.sum(head_relation_triplets == subject_relation, dim=1)
            delete_index = torch.nonzero(delete_index == 2).squeeze()

            delete_entity_index = all_triplets[delete_index, 2].view(-1).numpy()
            perturb_entity_index = np.array(list(set(np.arange(num_entity)) - set(delete_entity_index)))
            perturb_entity_index = torch.from_numpy(perturb_entity_index)
            perturb_entity_index = torch.cat((perturb_entity_index, object_.view(-1)))

            emb_ar = embedding[subject] * w[relation]
            emb_ar = emb_ar.view(-1, 1, 1)

            emb_c = embedding[perturb_entity_index]
            emb_c = emb_c.transpose(0, 1).unsqueeze(1)

            out_prod = torch.bmm(emb_ar, emb_c)
            score = torch.sum(out_prod, dim=0)
            score = torch.sigmoid(score)

            target = torch.tensor(len(perturb_entity_index) - 1)
            ranks_s.append(sort_and_rank(score, target))

            # Perturb subject
            object_ = test_triplet[2]
            relation = test_triplet[1]
            subject = test_triplet[0]

            object_relation = torch.tensor([object_, relation])
            delete_index = torch.sum(tail_relation_triplets == object_relation, dim=1)
            delete_index = torch.nonzero(delete_index == 2).squeeze()

            delete_entity_index = all_triplets[delete_index, 0].view(-1).numpy()
            perturb_entity_index = np.array(list(set(np.arange(num_entity)) - set(delete_entity_index)))
            perturb_entity_index = torch.from_numpy(perturb_entity_index)
            perturb_entity_index = torch.cat((perturb_entity_index, subject.view(-1)))

            emb_ar = embedding[object_] * w[relation]
            emb_ar = emb_ar.view(-1, 1, 1)

            emb_c = embedding[perturb_entity_index]
            emb_c = emb_c.transpose(0, 1).unsqueeze(1)

            out_prod = torch.bmm(emb_ar, emb_c)
            score = torch.sum(out_prod, dim=0)
            score = torch.sigmoid(score)

            target = torch.tensor(len(perturb_entity_index) - 1)
            ranks_o.append(sort_and_rank(score, target))

        ranks_s = torch.cat(ranks_s)
        ranks_o = torch.cat(ranks_o)

        ranks = torch.cat([ranks_s, ranks_o])
        ranks += 1  # change to 1-indexed

        mrr = torch.mean(1.0 / ranks.float())
        print("MRR (filtered): {:.6f}".format(mrr.item()))

        for hit in hits:
            avg_count = torch.mean((ranks <= hit).float())
            print("Hits (filtered) @ {}: {:.6f}".format(hit, avg_count.item()))

    return mrr.item()


# new code is below
def _infer_id_and_name(parts):

    if len(parts) != 2:
        raise ValueError(f"Expected exactly 2 columns in mapping file, got {len(parts)}: {parts}")

    a, b = parts

    def is_int(x):
        try:
            int(x)
            return True
        except ValueError:
            return False

    a_is_int = is_int(a)
    b_is_int = is_int(b)

    if a_is_int and not b_is_int:
        return b, int(a)
    elif b_is_int and not a_is_int:
        return a, int(b)
    else:
        raise ValueError(
            f"Cannot infer mapping (which is id, which is name) from: {parts}. "
            "One column must be an integer id and the other a string."
        )


def load_entity_relation_mappings(entities_path, relations_path):

    entity2id = {}
    relation2id = {}

    # Entities
    with open(entities_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            name, idx = _infer_id_and_name(parts)
            if name in entity2id and entity2id[name] != idx:
                raise ValueError(f"Inconsistent id for entity {name}: {entity2id[name]} vs {idx}")
            entity2id[name] = idx

    # Relations
    with open(relations_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            name, idx = _infer_id_and_name(parts)
            if name in relation2id and relation2id[name] != idx:
                raise ValueError(f"Inconsistent id for relation {name}: {relation2id[name]} vs {idx}")
            relation2id[name] = idx

    return entity2id, relation2id


def load_kegg_triples_single_file(kegg_path, entity2id, relation2id):

    triplets = []

    with open(kegg_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 3:
                raise ValueError(
                    f"KEGG triple file '{kegg_path}' line {line_idx + 1}: expected 3 columns, got {len(parts)}"
                )
            h_str, r_str, t_str = parts

            if h_str not in entity2id:
                raise KeyError(f"Head entity '{h_str}' (line {line_idx + 1}) not found in entities2id mapping.")
            if t_str not in entity2id:
                raise KeyError(f"Tail entity '{t_str}' (line {line_idx + 1}) not found in entities2id mapping.")
            if r_str not in relation2id:
                raise KeyError(f"Relation '{r_str}' (line {line_idx + 1}) not found in relations2id mapping.")

            h = entity2id[h_str]
            r = relation2id[r_str]
            t = entity2id[t_str]
            triplets.append((h, r, t))

    if len(triplets) == 0:
        raise ValueError(f"No triples loaded from KEGG file: {kegg_path}")

    #     # ---------- SAVE BEFORE RETURN ----------
    # os.makedirs("triplets", exist_ok=True)
    # out_path = os.path.join("triplets", "kegg_triplets_id.txt")
    #
    # with open(out_path, "w", encoding="utf-8") as f:
    #      for h, r, t in triplets:
    #         f.write(f"{h}\t{r}\t{t}\n")
    #     # --------------------------------------
    # exit()

    return np.array(triplets, dtype=np.int64)


def load_ddi_pairs(ddi_path, entity2id):

    heads = []
    tails = []
    labels = []

    with open(ddi_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 3:
                raise ValueError(
                    f"DDI file '{ddi_path}' line {line_idx + 1}: expected 3 columns (e1 e2 label), got {len(parts)}"
                )

            e1_str, e2_str, label_str = parts

            if e1_str not in entity2id:
                raise KeyError(f"DDI head entity '{e1_str}' (line {line_idx + 1}) not in entities2id mapping.")
            if e2_str not in entity2id:
                raise KeyError(f"DDI tail entity '{e2_str}' (line {line_idx + 1}) not in entities2id mapping.")

            try:
                label = int(label_str)
            except ValueError:
                raise ValueError(
                    f"DDI label '{label_str}' (line {line_idx + 1}) is not an integer 0/1."
                )

            if label not in (0, 1):
                raise ValueError(
                    f"DDI label '{label_str}' (line {line_idx + 1}) must be 0 or 1."
                )

            heads.append(entity2id[e1_str])
            tails.append(entity2id[e2_str])
            labels.append(label)

    if len(heads) == 0:
        raise ValueError(f"No DDI pairs loaded from file: {ddi_path}")

    return (
        np.array(heads, dtype=np.int64),
        np.array(tails, dtype=np.int64),
        np.array(labels, dtype=np.int64),
    )


def load_ddi_splits_from_csv(train_path, valid_path, test_path, entity2id, delimiter=","):

    # Infer the valid range of integer entity ids from the mapping
    if not entity2id:
        raise ValueError("entity2id mapping is empty.")
    max_entity_id = max(entity2id.values())

    def _to_entity_id(token, path, line_idx):
        token = token.strip()

        # Case 1: token looks like an integer; try to interpret as an id
        try:
            idx = int(token)
            if 0 <= idx <= max_entity_id:
                return idx
        except ValueError:
            pass

        # Case 2: token should be a name key in entity2id
        if token in entity2id:
            return entity2id[token]

        # If neither works, raise a clear error
        raise KeyError(
            f"DDI entity '{token}' (file '{path}', line {line_idx + 1}) "
            f"not found as an integer id in [0,{max_entity_id}] nor as a name in entities2id."
        )

    def _load_one(path):
        heads = []
        tails = []
        labels = []
        with open(path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                parts = line.split(delimiter)
                if len(parts) != 3:
                    raise ValueError(
                        f"File '{path}' line {line_idx + 1}: expected 3 columns "
                        f"(e1{delimiter}e2{delimiter}label), got {len(parts)}."
                    )
                e1_str = parts[0].strip()
                e2_str = parts[1].strip()
                label_str = parts[2].strip()

                # Map entities to integer indices (auto-detect id vs name)
                h_id = _to_entity_id(e1_str, path, line_idx)
                t_id = _to_entity_id(e2_str, path, line_idx)

                # Parse label
                try:
                    label = int(label_str)
                except ValueError:
                    raise ValueError(
                        f"DDI label '{label_str}' (file '{path}', line {line_idx + 1}) "
                        f"is not an integer 0/1."
                    )
                if label not in (0, 1):
                    raise ValueError(
                        f"DDI label '{label_str}' (file '{path}', line {line_idx + 1}) "
                        f"must be 0 or 1."
                    )

                heads.append(h_id)
                tails.append(t_id)
                labels.append(label)

        if len(heads) == 0:
            raise ValueError(f"No DDI pairs loaded from file: {path}")

        return (
            np.array(heads, dtype=np.int64),
            np.array(tails, dtype=np.int64),
            np.array(labels, dtype=np.int64),
        )

    h_train, t_train, y_train = _load_one(train_path)
    h_valid, t_valid, y_valid = _load_one(valid_path)
    h_test, t_test, y_test = _load_one(test_path)

    return (h_train, t_train, y_train), (h_valid, t_valid, y_valid), (h_test, t_test, y_test)


def build_kegg_adjacency(kegg_triplets, num_entities, max_degree: int = 20):  # how many neighbours to keep 5 etc

    # 1. Build full undirected adjacency
    adjacency = [[] for _ in range(num_entities)]

    for h, r, t in kegg_triplets:
        h = int(h)
        t = int(t)
        # Undirected neighbor view for sampling (RGCN will get directed edges separately)
        adjacency[h].append(t)
        adjacency[t].append(h)

    # 2. Optionally apply degree cap (GraphSAGE-style)
    if max_degree is not None:
        for u in range(num_entities):
            neighs = adjacency[u]
            if len(neighs) > max_degree:
                # Random but reproducible because we set global seeds in ddi_main.py
                adjacency[u] = random.sample(neighs, max_degree)
    return adjacency




import random
from typing import Any, Dict
import hashlib
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborSampler  # kept, but base sampling below is CSR-exactK

# ------------------------------------------------------------
# GLOBAL CACHE
# ------------------------------------------------------------
_PYG_CACHE = {
    "full_data": None,
    "samplers": {},          # fanout_tuple -> NeighborSampler (optional for other uses)
    "num_entities": None,
    "num_relations": None,
    "edge_index_id": None,
    "csr": None,             # <-- CSR for exact-K sampling
}


def _get_full_data_and_sampler(kegg_triplets, num_entities, num_relations, fanouts_per_hop):
    global _PYG_CACHE

    fanouts_key = tuple(int(x) for x in fanouts_per_hop)

    same_full = (
        _PYG_CACHE["full_data"] is not None and
        _PYG_CACHE["num_entities"] == int(num_entities) and
        _PYG_CACHE["num_relations"] == int(num_relations) and
        _PYG_CACHE["edge_index_id"] == id(kegg_triplets)
    )

    if not same_full:
        inv_offset = int(num_relations)

        src = []
        dst = []
        ety = []

        # Build FULL KG: forward + inverse
        for (h, r, t) in kegg_triplets:
            h = int(h); r = int(r); t = int(t)
            src.append(h); dst.append(t); ety.append(r)                 # forward
            src.append(t); dst.append(h); ety.append(r + inv_offset)    # inverse

        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_type  = torch.tensor(ety, dtype=torch.long)

        full_data = Data(edge_index=edge_index, edge_type=edge_type, num_nodes=int(num_entities))

        # ---------------- CSR for EXACT-K sampling (outgoing edges) ----------------
        # Sort edges by src to build rowptr/col/eid
        src_t = edge_index[0]
        dst_t = edge_index[1]
        eids  = torch.arange(src_t.numel(), dtype=torch.long)

        perm = torch.argsort(src_t)
        src_s = src_t[perm]
        dst_s = dst_t[perm]
        eid_s = eids[perm]

        deg = torch.bincount(src_s, minlength=int(num_entities))
        rowptr = torch.empty(int(num_entities) + 1, dtype=torch.long)
        rowptr[0] = 0
        rowptr[1:] = torch.cumsum(deg, dim=0)

        _PYG_CACHE["csr"] = {
            "rowptr": rowptr,   # [N+1]
            "col": dst_s,       # [E]
            "eid": eid_s,       # [E] indexes into full_data.edge_type
        }
        # -------------------------------------------------------------------------

        _PYG_CACHE["full_data"] = full_data
        _PYG_CACHE["num_entities"] = int(num_entities)
        _PYG_CACHE["num_relations"] = int(num_relations)
        _PYG_CACHE["edge_index_id"] = id(kegg_triplets)
        _PYG_CACHE["samplers"] = {}  # reset samplers when full graph changes
        # _PYG_CACHE["samplers"].clear()

    full_data = _PYG_CACHE["full_data"]

    # keep your sampler cache (optional; not used for exactK base sampling below)
    sampler = _PYG_CACHE["samplers"].get(fanouts_key)
    if sampler is None:
        sampler = NeighborSampler(
            full_data.edge_index,
            list(fanouts_key),
            batch_size=1,
            shuffle=False,
            num_nodes=int(num_entities),
        )
        _PYG_CACHE["samplers"][fanouts_key] = sampler

    return full_data, sampler


def build_ddi_subgraph_random_view1_view2_different(
        batch_heads,
        batch_tails,
        adjacency,       # unused
        kegg_triplets,
        edges_per_node,  # unused
        num_entities,
        num_relations,
        # neigh_2,
        # neigh_3,
        # device,
        id2row2, vals2, id2row3, vals3,epoch
):
    # ------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------
    fanouts_orig = (6, 5)  # EXACT: hop1=10, hop2=5 (unless degree smaller)
    fanouts_aug = (5,)  # EXACT tail context: hop1=5 (unless degree smaller)

    num_aug_per_drug = 4

    # if epoch == 1:
    #     num_aug_per_drug = 4
    # elif 1 < epoch < 50:
    #     num_aug_per_drug = 0
    # else:
    #     num_aug_per_drug = 4

    #default 8
    triadic_rel_id = 42
    l3_rel_id = 43

    inv_offset = int(num_relations)
    num_rel_total = int(num_relations) * 2

    # ------------------------------------------------------------
    # 1) Target seeds
    # ------------------------------------------------------------
    target_nodes = sorted(set(int(x) for x in batch_heads) | set(int(x) for x in batch_tails))
    seed_orig = torch.tensor(target_nodes, dtype=torch.long)




    # ------------------------------------------------------------
    # full graph + (optional) samplers
    # ------------------------------------------------------------
    full_data, sampler_orig = _get_full_data_and_sampler(
        kegg_triplets=kegg_triplets,
        num_entities=num_entities,
        num_relations=num_relations,
        fanouts_per_hop=fanouts_orig,
    )
    _, sampler_aug = _get_full_data_and_sampler(
        kegg_triplets=kegg_triplets,
        num_entities=num_entities,
        num_relations=num_relations,
        fanouts_per_hop=fanouts_aug,
    )

    # ------------------------------------------------------------
    # EXACT-K sampling via CSR (NO PyG NeighborSampler semantics)
    # ------------------------------------------------------------
    @torch.no_grad()
    def sample_subgraph_exact(seed_nodes, fanouts_per_hop):
        """
        Exact-K per node per hop:
          hop1: each seed u picks exactly min(K1, outdeg(u)) outgoing edges
          hop2: each frontier v picks exactly min(K2, outdeg(v)) outgoing edges
        Returns:
          n_id (GLOBAL node ids, sorted),
          edge_index (LOCAL indices into n_id),
          edge_type (relation ids for each edge)
        """
        if seed_nodes.numel() == 0:
            return (
                torch.empty((0,), dtype=torch.long),
                torch.empty((2, 0), dtype=torch.long),
                torch.empty((0,), dtype=torch.long),
            )

        csr = _PYG_CACHE["csr"]
        rowptr = csr["rowptr"]
        col    = csr["col"]
        eidtab = csr["eid"]

        K1 = int(fanouts_per_hop[0]) if len(fanouts_per_hop) > 0 else 0
        K2 = int(fanouts_per_hop[1]) if len(fanouts_per_hop) > 1 else 0

        seeds = torch.unique(seed_nodes.to(torch.long).cpu())

        src_all = []
        dst_all = []
        eid_all = []

        def pick_edges_for_nodes(nodes, K):
            if K <= 0 or nodes.numel() == 0:
                return None, None, None
            out_src = []
            out_dst = []
            out_eid = []
            for u in nodes.tolist():
                if u < 0 or u >= int(num_entities):
                    continue
                start = int(rowptr[u].item())
                end   = int(rowptr[u + 1].item())
                deg   = end - start
                if deg <= 0:
                    continue
                take = deg if deg < K else K
                if deg <= K:
                    idx = torch.arange(start, end, dtype=torch.long)
                else:
                    idx = start + torch.randperm(deg)[:take]  # exact take without replacement
                out_src.append(torch.full((idx.numel(),), u, dtype=torch.long))
                out_dst.append(col[idx].to(torch.long))
                out_eid.append(eidtab[idx].to(torch.long))
            if not out_src:
                return None, None, None
            return torch.cat(out_src, dim=0), torch.cat(out_dst, dim=0), torch.cat(out_eid, dim=0)

        # hop 1
        src1, dst1, eid1 = pick_edges_for_nodes(seeds, K1)
        if src1 is not None:
            src_all.append(src1); dst_all.append(dst1); eid_all.append(eid1)
            frontier = torch.unique(dst1)
        else:
            frontier = torch.empty((0,), dtype=torch.long)

        # hop 2
        if K2 > 0 and frontier.numel() > 0:
            src2, dst2, eid2 = pick_edges_for_nodes(frontier, K2)
            if src2 is not None:
                src_all.append(src2); dst_all.append(dst2); eid_all.append(eid2)

        if not src_all:
            n_id = torch.unique(seeds, sorted=True)
            return n_id, torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.long)

        src_g = torch.cat(src_all, dim=0)
        dst_g = torch.cat(dst_all, dim=0)
        e_id  = torch.cat(eid_all, dim=0)

        n_id = torch.unique(torch.cat([seeds, src_g, dst_g], dim=0), sorted=True)

        src_l = torch.searchsorted(n_id, src_g)
        dst_l = torch.searchsorted(n_id, dst_g)
        edge_index = torch.stack([src_l, dst_l], dim=0)

        edge_type = full_data.edge_type[e_id]
        return n_id, edge_index, edge_type

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------
    def local_to_global(entity, edge_index, edge_type):
        if edge_type.numel() == 0:
            dev = entity.device
            return (
                torch.empty((0,), dtype=torch.long, device=dev),
                torch.empty((0,), dtype=torch.long, device=dev),
                torch.empty((0,), dtype=torch.long, device=dev),
            )
        h = entity[edge_index[0]]
        t = entity[edge_index[1]]
        r = edge_type
        return h, r, t

    def pair_and_dedup(h, r, t):
        if r.numel() == 0:
            dev = r.device
            return (
                torch.empty((0,), dtype=torch.long, device=dev),
                torch.empty((0,), dtype=torch.long, device=dev),
                torch.empty((0,), dtype=torch.long, device=dev),
            )

        r_f = r % inv_offset
        is_fwd = (r < inv_offset)

        h_f = torch.where(is_fwd, h, t)
        t_f = torch.where(is_fwd, t, h)

        N = int(num_entities)
        key = h_f * (inv_offset * N) + r_f * N + t_f

        perm = torch.argsort(key)
        key_s = key[perm]

        keep = torch.ones_like(key_s, dtype=torch.bool)
        keep[1:] = key_s[1:] != key_s[:-1]

        idx = perm[keep]
        h_f, r_f, t_f = h_f[idx], r_f[idx], t_f[idx]

        h2 = torch.cat([h_f, t_f], dim=0)
        t2 = torch.cat([t_f, h_f], dim=0)
        r2 = torch.cat([r_f, r_f + inv_offset], dim=0)

        key2 = h2 * (num_rel_total * N) + r2 * N + t2
        perm2 = torch.argsort(key2)
        key2s = key2[perm2]

        keep2 = torch.ones_like(key2s, dtype=torch.bool)
        keep2[1:] = key2s[1:] != key2s[:-1]

        idx2 = perm2[keep2]
        return h2[idx2], r2[idx2], t2[idx2]

    def build_data(node_list, h, r, t):
        nodes = torch.tensor(node_list, dtype=torch.long)
        h_l = torch.searchsorted(nodes, h)
        t_l = torch.searchsorted(nodes, t)

        edge_index = torch.stack([h_l, t_l], dim=0)
        edge_type = r

        edge_norm = edge_normalization(
            edge_type=edge_type,
            edge_index=edge_index,
            num_entity=len(node_list),
            num_relation=num_rel_total,
        )

        g2l = {int(g): i for i, g in enumerate(node_list)}

        return Data(
            entity=nodes,
            edge_index=edge_index,
            edge_type=edge_type,
            edge_norm=edge_norm,
        ), g2l

    # ------------------------------------------------------------
    # AUG edges from neigh tables (unchanged)
    # ------------------------------------------------------------
    def aug_edges_firstk_fixedK(target_nodes, id2row, vals, rel_id: int, k: int):
        u = torch.as_tensor(target_nodes, dtype=torch.long, device="cpu").view(-1)

        m = (u >= 0) & (u < id2row.numel())
        if not bool(m.any()):
            return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.long)

        u = u[m]
        rows = id2row[u].to(torch.long)
        keep = rows != -1
        if not bool(keep.any()):
            return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.long)

        u = u[keep]
        rows = rows[keep]

        k = int(k)
        if k <= 0:
            return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.long)

        k_eff = min(k, vals.size(1))
        picks = vals[rows, :k_eff]
        mask = picks != -1
        if not bool(mask.any()):
            return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.long)

        dst = picks[mask].to(torch.long)
        cnt = mask.sum(dim=1).to(torch.long)
        src = u.repeat_interleave(cnt)

        edge_index = torch.stack([src, dst], dim=0)
        edge_type = torch.full((dst.numel(),), int(rel_id), dtype=torch.long)
        return edge_index, edge_type

    ei_tri, et_tri = aug_edges_firstk_fixedK(target_nodes, id2row2, vals2, triadic_rel_id, num_aug_per_drug)
    ei_l3,  et_l3  = aug_edges_firstk_fixedK(target_nodes, id2row3, vals3, l3_rel_id, num_aug_per_drug)

    edge_index_aug = torch.cat([ei_tri, ei_l3], dim=1)
    edge_type_aug  = torch.cat([et_tri, et_l3], dim=0)

    tails_a = set(edge_index_aug[1].tolist())  # augmented-neighbor nodes (GLOBAL ids)

    # ------------------------------------------------------------
    # Build views
    # ------------------------------------------------------------
    def build_view(extra_edge_index, extra_edge_type, tail_nodes, fanouts_tgt, fanouts_tail):
        # ---- TARGET SAMPLING (EXACT-K) ----
        ent_tgt, ei_tgt, et_tgt = sample_subgraph_exact(seed_orig, fanouts_tgt)
        h_tgt, r_tgt, t_tgt = local_to_global(ent_tgt, ei_tgt, et_tgt)

        # ---- tail context (EXACT-K) ----
        if tail_nodes:
            seed_tail = torch.tensor(list(tail_nodes), dtype=torch.long)
            ent_tail, ei_tail, et_tail = sample_subgraph_exact(seed_tail, fanouts_tail)
            h_tail, r_tail, t_tail = local_to_global(ent_tail, ei_tail, et_tail)
            tail_ctx = set(int(x) for x in ent_tail.tolist())
        else:
            h_tail = r_tail = t_tail = torch.empty((0,), dtype=torch.long)
            tail_ctx = set()

        nodes = set(ent_tgt.tolist()) | set(tail_nodes) | set(tail_ctx)
        node_list = sorted(nodes)

        # ---- extra aug edges are GLOBAL tensors ----
        if extra_edge_type is not None and extra_edge_type.numel() > 0:
            h_ex = extra_edge_index[0].to(torch.long)
            t_ex = extra_edge_index[1].to(torch.long)
            r_ex = extra_edge_type.to(torch.long)
        else:
            h_ex = r_ex = t_ex = torch.empty((0,), dtype=torch.long)



  #Below few lines are if we want inverse on all real and aug edges
        h_all = torch.cat([h_tgt, h_tail, h_ex], dim=0)
        r_all = torch.cat([r_tgt, r_tail, r_ex], dim=0)
        t_all = torch.cat([t_tgt, t_tail, t_ex], dim=0)

        h_all, r_all, t_all = pair_and_dedup(h_all, r_all, t_all)
        return build_data(node_list, h_all, r_all, t_all)

    # ORG: exact (10,5) around seeds, no tail context
    # data_orig, g2l_orig = build_view(
    #     extra_edge_index=None,
    #     extra_edge_type=None,
    #     tail_nodes=[],
    #     fanouts_tgt=fanouts_orig,
    #     fanouts_tail=fanouts_aug,
    # )

    # OR use if we want to use the augmented edges on the org graph as well
    data_orig, g2l_orig = build_view(
        extra_edge_index=edge_index_aug,
        extra_edge_type=edge_type_aug,
        tail_nodes=tails_a,
        fanouts_tgt=fanouts_orig,
        fanouts_tail=fanouts_aug,
    )

    # AUG: exact (10,5) around seeds + exact (5,) around tails_a + add augmented edges
    data_aug, g2l_aug = build_view(
        extra_edge_index=edge_index_aug,
        extra_edge_type=edge_type_aug,
        tail_nodes=tails_a,
        fanouts_tgt=fanouts_orig,
        fanouts_tail=fanouts_aug,
    )

    # ------------------------------------------------------------
    # Convert GLOBAL seeds -> LOCAL ids (preserve order)
    # ------------------------------------------------------------
    # seed_local_orig = []  # keep your behavior
    # seed_local_aug = []  # keep your behaviour
    seed_local_orig = torch.tensor(
        [g2l_orig[g] for g in target_nodes if g in g2l_orig],
        dtype=torch.long
    )
    seed_local_aug = torch.tensor(
        [g2l_aug[g] for g in target_nodes if g in g2l_aug],
        dtype=torch.long
    )
    return data_orig, data_aug, g2l_orig, g2l_aug, seed_local_orig, seed_local_aug


def build_ddi_subgraph_random_view1_view2_different_eval(
        batch_heads,
        batch_tails,
        adjacency,       # unused
        kegg_triplets,
        edges_per_node,  # unused
        num_entities,
        num_relations,
        # neigh_2,
        # neigh_3,
        # device,
        id2row2, vals2, id2row3, vals3, epoch
):
    # ------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------
    fanouts_orig = (6, 5)  # EXACT: hop1=10, hop2=5 (unless degree smaller)
    fanouts_aug = (5,)  # EXACT tail context: hop1=5 (unless degree smaller)
    num_aug_per_drug = 4

    # if epoch == 1:
    #     num_aug_per_drug = 4
    # elif 1 < epoch < 50:
    #     num_aug_per_drug = 0
    # else:
    #     num_aug_per_drug = 4

    triadic_rel_id = 42
    l3_rel_id = 43

    inv_offset = int(num_relations)
    num_rel_total = int(num_relations) * 2

    # ------------------------------------------------------------
    # 1) Target seeds
    # ------------------------------------------------------------
    target_nodes = sorted(set(int(x) for x in batch_heads) | set(int(x) for x in batch_tails))
    seed_orig = torch.tensor(target_nodes, dtype=torch.long)


    # ------------------------------------------------------------
    # full graph + (optional) samplers
    # ------------------------------------------------------------
    full_data, sampler_orig = _get_full_data_and_sampler(
        kegg_triplets=kegg_triplets,
        num_entities=num_entities,
        num_relations=num_relations,
        fanouts_per_hop=fanouts_orig,
    )
    _, sampler_aug = _get_full_data_and_sampler(
        kegg_triplets=kegg_triplets,
        num_entities=num_entities,
        num_relations=num_relations,
        fanouts_per_hop=fanouts_aug,
    )

    # ------------------------------------------------------------
    # EXACT-K sampling via CSR (NO PyG NeighborSampler semantics)
    # ------------------------------------------------------------
    @torch.no_grad()
    def sample_subgraph_exact(seed_nodes, fanouts_per_hop, view_id: int, base_seed: int):
        """
        Exact-K per node per hop:
          hop1: each seed u picks exactly min(K1, outdeg(u)) outgoing edges
          hop2: each frontier v picks exactly min(K2, outdeg(v)) outgoing edges
        Returns:
          n_id (GLOBAL node ids, sorted),
          edge_index (LOCAL indices into n_id),
          edge_type (relation ids for each edge)
        """
        if seed_nodes.numel() == 0:
            return (
                torch.empty((0,), dtype=torch.long),
                torch.empty((2, 0), dtype=torch.long),
                torch.empty((0,), dtype=torch.long),
            )

        csr = _PYG_CACHE["csr"]
        rowptr = csr["rowptr"]
        col    = csr["col"]
        eidtab = csr["eid"]

        K1 = int(fanouts_per_hop[0]) if len(fanouts_per_hop) > 0 else 0
        K2 = int(fanouts_per_hop[1]) if len(fanouts_per_hop) > 1 else 0

        seeds = torch.unique(seed_nodes.to(torch.long).cpu())

        src_all = []
        dst_all = []
        eid_all = []

        def pick_edges_for_nodes(nodes, K, hop_id: int, view_id: int, base_seed: int):
            if K <= 0 or nodes.numel() == 0:
                return None, None, None
            out_src = []
            out_dst = []
            out_eid = []
            for u in nodes.tolist():
                if u < 0 or u >= int(num_entities):
                    continue
                start = int(rowptr[u].item())
                end   = int(rowptr[u + 1].item())
                deg   = end - start
                if deg <= 0:
                    continue
                take = deg if deg < K else K
                if deg <= K:
                    idx = torch.arange(start, end, dtype=torch.long)
                else:
                    seed_str = f"{base_seed}|{view_id}|{hop_id}|{u}"
                    seed = int.from_bytes(hashlib.md5(seed_str.encode()).digest()[:8], "little") % (2 ** 63 - 1)
                    g = torch.Generator(device="cpu")
                    g.manual_seed(seed)
                    idx = start + torch.randperm(deg, generator=g)[:take]  # deterministic per node/hop/view
                    # exact take without replacement
                out_src.append(torch.full((idx.numel(),), u, dtype=torch.long))
                out_dst.append(col[idx].to(torch.long))
                out_eid.append(eidtab[idx].to(torch.long))
            if not out_src:
                return None, None, None
            return torch.cat(out_src, dim=0), torch.cat(out_dst, dim=0), torch.cat(out_eid, dim=0)

        # hop 1
        src1, dst1, eid1 = pick_edges_for_nodes(seeds, K1, hop_id=1, view_id=view_id, base_seed=base_seed)
        if src1 is not None:
            src_all.append(src1); dst_all.append(dst1); eid_all.append(eid1)
            frontier = torch.unique(dst1)
        else:
            frontier = torch.empty((0,), dtype=torch.long)

        # hop 2
        if K2 > 0 and frontier.numel() > 0:
            src2, dst2, eid2 = pick_edges_for_nodes(frontier, K2, hop_id=2, view_id=view_id, base_seed=base_seed)
            if src2 is not None:
                src_all.append(src2); dst_all.append(dst2); eid_all.append(eid2)

        if not src_all:
            n_id = torch.unique(seeds, sorted=True)
            return n_id, torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.long)

        src_g = torch.cat(src_all, dim=0)
        dst_g = torch.cat(dst_all, dim=0)
        e_id  = torch.cat(eid_all, dim=0)

        n_id = torch.unique(torch.cat([seeds, src_g, dst_g], dim=0), sorted=True)

        src_l = torch.searchsorted(n_id, src_g)
        dst_l = torch.searchsorted(n_id, dst_g)
        edge_index = torch.stack([src_l, dst_l], dim=0)

        edge_type = full_data.edge_type[e_id]
        return n_id, edge_index, edge_type

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------
    def local_to_global(entity, edge_index, edge_type):
        if edge_type.numel() == 0:
            dev = entity.device
            return (
                torch.empty((0,), dtype=torch.long, device=dev),
                torch.empty((0,), dtype=torch.long, device=dev),
                torch.empty((0,), dtype=torch.long, device=dev),
            )
        h = entity[edge_index[0]]
        t = entity[edge_index[1]]
        r = edge_type
        return h, r, t

    def pair_and_dedup(h, r, t):
        if r.numel() == 0:
            dev = r.device
            return (
                torch.empty((0,), dtype=torch.long, device=dev),
                torch.empty((0,), dtype=torch.long, device=dev),
                torch.empty((0,), dtype=torch.long, device=dev),
            )

        r_f = r % inv_offset
        is_fwd = (r < inv_offset)

        h_f = torch.where(is_fwd, h, t)
        t_f = torch.where(is_fwd, t, h)

        N = int(num_entities)
        key = h_f * (inv_offset * N) + r_f * N + t_f

        perm = torch.argsort(key)
        key_s = key[perm]

        keep = torch.ones_like(key_s, dtype=torch.bool)
        keep[1:] = key_s[1:] != key_s[:-1]

        idx = perm[keep]
        h_f, r_f, t_f = h_f[idx], r_f[idx], t_f[idx]

        h2 = torch.cat([h_f, t_f], dim=0)
        t2 = torch.cat([t_f, h_f], dim=0)
        r2 = torch.cat([r_f, r_f + inv_offset], dim=0)

        key2 = h2 * (num_rel_total * N) + r2 * N + t2
        perm2 = torch.argsort(key2)
        key2s = key2[perm2]

        keep2 = torch.ones_like(key2s, dtype=torch.bool)
        keep2[1:] = key2s[1:] != key2s[:-1]

        idx2 = perm2[keep2]
        return h2[idx2], r2[idx2], t2[idx2]

    def build_data(node_list, h, r, t):
        nodes = torch.tensor(node_list, dtype=torch.long)
        h_l = torch.searchsorted(nodes, h)
        t_l = torch.searchsorted(nodes, t)

        edge_index = torch.stack([h_l, t_l], dim=0)
        edge_type = r

        edge_norm = edge_normalization(
            edge_type=edge_type,
            edge_index=edge_index,
            num_entity=len(node_list),
            num_relation=num_rel_total,
        )

        g2l = {int(g): i for i, g in enumerate(node_list)}

        return Data(
            entity=nodes,
            edge_index=edge_index,
            edge_type=edge_type,
            edge_norm=edge_norm,
        ), g2l

    # ------------------------------------------------------------
    # AUG edges from neigh tables (unchanged)
    # ------------------------------------------------------------
    def aug_edges_firstk_fixedK(target_nodes, id2row, vals, rel_id: int, k: int):
        u = torch.as_tensor(target_nodes, dtype=torch.long, device="cpu").view(-1)

        m = (u >= 0) & (u < id2row.numel())
        if not bool(m.any()):
            return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.long)

        u = u[m]
        rows = id2row[u].to(torch.long)
        keep = rows != -1
        if not bool(keep.any()):
            return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.long)

        u = u[keep]
        rows = rows[keep]

        k = int(k)
        if k <= 0:
            return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.long)

        k_eff = min(k, vals.size(1))
        picks = vals[rows, :k_eff]
        mask = picks != -1
        if not bool(mask.any()):
            return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.long)

        dst = picks[mask].to(torch.long)
        cnt = mask.sum(dim=1).to(torch.long)
        src = u.repeat_interleave(cnt)

        edge_index = torch.stack([src, dst], dim=0)
        edge_type = torch.full((dst.numel(),), int(rel_id), dtype=torch.long)
        return edge_index, edge_type

    ei_tri, et_tri = aug_edges_firstk_fixedK(target_nodes, id2row2, vals2, triadic_rel_id, num_aug_per_drug)
    ei_l3,  et_l3  = aug_edges_firstk_fixedK(target_nodes, id2row3, vals3, l3_rel_id, num_aug_per_drug)

    edge_index_aug = torch.cat([ei_tri, ei_l3], dim=1)
    edge_type_aug  = torch.cat([et_tri, et_l3], dim=0)

    tails_a = set(edge_index_aug[1].tolist())  # augmented-neighbor nodes (GLOBAL ids)

    # ------------------------------------------------------------
    # Build views
    # ------------------------------------------------------------
    def build_view(extra_edge_index, extra_edge_type, tail_nodes, fanouts_tgt, fanouts_tail, view_id: int, base_seed: int):
        # ---- TARGET SAMPLING (EXACT-K) ----
        ent_tgt, ei_tgt, et_tgt = sample_subgraph_exact(seed_orig, fanouts_tgt, view_id=view_id, base_seed=base_seed)
        h_tgt, r_tgt, t_tgt = local_to_global(ent_tgt, ei_tgt, et_tgt)

        # ---- tail context (EXACT-K) ----
        if tail_nodes:
            seed_tail = torch.tensor(list(tail_nodes), dtype=torch.long)
            ent_tail, ei_tail, et_tail = sample_subgraph_exact(seed_tail, fanouts_tail, view_id=view_id, base_seed=base_seed)
            h_tail, r_tail, t_tail = local_to_global(ent_tail, ei_tail, et_tail)
            tail_ctx = set(int(x) for x in ent_tail.tolist())
        else:
            h_tail = r_tail = t_tail = torch.empty((0,), dtype=torch.long)
            tail_ctx = set()

        nodes = set(ent_tgt.tolist()) | set(tail_nodes) | set(tail_ctx)
        node_list = sorted(nodes)

        # ---- extra aug edges are GLOBAL tensors ----
        if extra_edge_type is not None and extra_edge_type.numel() > 0:
            h_ex = extra_edge_index[0].to(torch.long)
            t_ex = extra_edge_index[1].to(torch.long)
            r_ex = extra_edge_type.to(torch.long)
        else:
            h_ex = r_ex = t_ex = torch.empty((0,), dtype=torch.long)

        h_all = torch.cat([h_tgt, h_tail, h_ex], dim=0)
        r_all = torch.cat([r_tgt, r_tail, r_ex], dim=0)
        t_all = torch.cat([t_tgt, t_tail, t_ex], dim=0)

        h_all, r_all, t_all = pair_and_dedup(h_all, r_all, t_all)
        return build_data(node_list, h_all, r_all, t_all)

    # ORG: exact (10,5) around seeds, no tail context
    # data_orig, g2l_orig = build_view(
    #     extra_edge_index=None,
    #     extra_edge_type=None,
    #     tail_nodes=[],
    #     fanouts_tgt=fanouts_orig,
    #     fanouts_tail=fanouts_aug,
    # )

    # OR use if we want to use the augmented edges on the org graph as well
    data_orig, g2l_orig = build_view(
        extra_edge_index=edge_index_aug,
        extra_edge_type=edge_type_aug,
        tail_nodes=tails_a,
        fanouts_tgt=fanouts_orig,
        fanouts_tail=fanouts_aug,
        view_id=0,
        base_seed=BASE_SEED,
    )

    # AUG: exact (10,5) around seeds + exact (5,) around tails_a + add augmented edges
    data_aug, g2l_aug = build_view(
        extra_edge_index=edge_index_aug,
        extra_edge_type=edge_type_aug,
        tail_nodes=tails_a,
        fanouts_tgt=fanouts_orig,
        fanouts_tail=fanouts_aug,
        view_id=1,
        base_seed=BASE_SEED,
    )

    # ------------------------------------------------------------
    # Convert GLOBAL seeds -> LOCAL ids (preserve order)
    # ------------------------------------------------------------
    # seed_local_orig = []  # keep your behavior
    # seed_local_aug = []  # keep your behaviour
    seed_local_orig = torch.tensor(
        [g2l_orig[g] for g in target_nodes if g in g2l_orig],
        dtype=torch.long
    )
    seed_local_aug = torch.tensor(
        [g2l_aug[g] for g in target_nodes if g in g2l_aug],
        dtype=torch.long
    )
    return data_orig, data_aug, g2l_orig, g2l_aug, seed_local_orig, seed_local_aug
