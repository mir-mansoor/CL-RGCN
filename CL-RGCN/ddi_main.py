
import os
os.environ["PYTHONHASHSEED"] = "1" # seed = 42
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # harmless on CPU, required if you ever use CUDA
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
from torch import nn
import torch.nn.functional as F

from torch.optim.lr_scheduler import ReduceLROnPlateau

from tqdm import tqdm
import pandas as pd


import os
import random

import numpy as np
import torch
from torch.backends import cudnn


def set_seed(seed: int):
    print("The current seed is below:")
    print(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
        torch.set_deterministic(True)
    except AttributeError:
        pass

from torch.utils.data import TensorDataset, DataLoader
import argparse

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, average_precision_score
from models import RGCN, DDIMLP




from utils import (
    load_entity_relation_mappings,
    load_kegg_triples_single_file,
    load_ddi_pairs,
    # split_ddi_dataset,
    load_ddi_splits_from_csv,  # <-- NEW
    build_kegg_adjacency,
    #build_ddi_subgraph_for_batch_eval,
    build_ddi_subgraph_random_view1_view2_different, build_ddi_subgraph_random_view1_view2_different_eval,

    # build_ddi_subgraph_for_batch_drugs_bi_directional, build_ddi_subgraph_for_batch_drugs_bi_directional_same_original,
    # build_ddi_subgraph_for_batch_drugs_bi_directional_same_original,
    # build_ddi_subgraph_for_batch_drugs_bi_directional_same_original2,
    # prepare_kg_gpu, build_ddi_subgraph_for_batch_drugs_bi_directional_same_original3,
    # build_ddi_subgraph_for_batch_drugs_bi_directional_same_original3,

)

# def save_full_test_set_with_probabilities(probs):
#     test_csv_path = "./data/DRKG/test.csv"
#     df = pd.read_csv(test_csv_path)
#
#     if len(probs) == len(df) + 1:
#         probs = probs[:len(df)]
#     elif len(probs) != len(df):
#         raise ValueError(
#             f"Row mismatch: test CSV has {len(df)} rows, but probabilities have {len(probs)} rows."
#         )
#
#     df["predicted_probability"] = probs.astype(float)
#     df["predicted_class"] = (probs >= 0.5).astype(int)
#
#     base, ext = os.path.splitext(test_csv_path)
#     output_file = f"{base}_with_probabilities.csv"
#
#     df.to_csv(output_file, index=False)
#     print(f"Saved full test set with probabilities to: {output_file}")

def save_full_test_set_with_probabilities(heads, tails, labels, probs):

    df = pd.DataFrame({
        "drug_id": heads,
        "interacting_drug": tails,
        "true_label": labels,
        "predicted_probability": probs,
        "predicted_class": (probs >= 0.5).astype(int)
    })

    output_file = "./data/DRKG/test_with_probabilities.csv"
    df.to_csv(output_file, index=False)

    print(f"Saved full test set with probabilities to: {output_file}")


def save_positive_pairs_for_drug(drug_id, heads, tails, labels, probs, output_file):
    rows = []

    for h, t, y, p in zip(heads, tails, labels, probs):

        if y == 1 and (h == drug_id or t == drug_id):

            other = t if h == drug_id else h

            rows.append({
                "drug_id": drug_id,
                "interacting_drug": int(other),
                "true_label": int(y),
                "predicted_probability": float(p),
                "predicted_class": int(p >= 0.5)
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)

    print(f"Saved {len(rows)} test interactions for drug {drug_id}")


#usinf skilearn for matrics
def compute_classification_metrics(y_true, y_prob, threshold=0.5):

    # Binary predictions at given threshold
    y_pred = (y_prob >= threshold).astype(np.int64)

    # Accuracy
    acc = accuracy_score(y_true, y_pred)

    # Precision, Recall, F1 (binary)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )

    # ROC-AUC (on probabilities)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        # e.g. only one class present in y_true
        auc = 0.0

    # AUPR (Average Precision)
    try:
        aupr = average_precision_score(y_true, y_prob)
    except ValueError:
        aupr = 0.0

    return {
        "acc": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc),
        "aupr": float(aupr),
    }


#def train_epoch_ddi(gnn, mlp, graph_data, data_loader, device, bce_loss_fn, optimizer):
def train_epoch_ddi_augmented(
        gnn,
        mlp,
        kegg_triplets,
        adjacency,
        edges_per_node,
        num_entities,
        num_relations,
        data_loader,
        device,
        bce_loss_fn,
        optimizer,
        epoch,
        id2row2,
        vals2,
        id2row3,
        vals3,

):
    import numpy as np

    import torch

    gnn.train()
    mlp.train()

    total_loss = 0.0
    num_batches = 0

    for (h_batch, t_batch, y_batch) in tqdm(
            data_loader,
            desc=f"Epoch {epoch}",
            unit="batch",
            leave=False,
    ):

        h_cpu = h_batch.numpy()
        t_cpu = t_batch.numpy()

        # Only labels need GPU for loss
        y_batch = y_batch.to(device).float()

        data_orig, data_aug1, global_to_local_orig, global_to_local_aug, seed_local_orig, seed_local_aug =\
            build_ddi_subgraph_random_view1_view2_different(
                batch_heads=h_cpu,
                batch_tails=t_cpu,
                adjacency=adjacency,  # precomputed with max_degree=20
                kegg_triplets=kegg_triplets,
                edges_per_node=edges_per_node,
                num_entities=num_entities,
                num_relations=num_relations,
                # neigh_2=neigh_2,
                # neigh_3=neigh_3,
                # device=device,
                id2row2=id2row2, vals2=vals2, id2row3=id2row3, vals3=vals3,epoch=epoch,
            )

        # Move subgraphs to device
        data_orig = data_orig.to(device)
        data_aug1 = data_aug1.to(device)

        # ---- RGCN forward on each view ----

        emb_aug1 = gnn(
            data_aug1.entity,
            data_aug1.edge_index,
            data_aug1.edge_type,
            data_aug1.edge_norm,
            debug_seed_nodes=seed_local_aug,
        )

        emb_orig = gnn(
            data_orig.entity,
            data_orig.edge_index,
            data_orig.edge_type,
            data_orig.edge_norm,
            debug_seed_nodes=seed_local_orig,
        )

        # ---- Map batch DDI entities to local indices in EACH view ----
        # original mapping
        h_local_idx_orig = torch.tensor(
            [global_to_local_orig[int(h)] for h in h_cpu],
            dtype=torch.long,
            device=device,
        )
        t_local_idx_orig = torch.tensor(
            [global_to_local_orig[int(t)] for t in t_cpu],
            dtype=torch.long,
            device=device,
        )

        # augmented mapping
        h_local_idx_aug = torch.tensor(
            [global_to_local_aug[int(h)] for h in h_cpu],
            dtype=torch.long,
            device=device,
        )
        t_local_idx_aug = torch.tensor(
            [global_to_local_aug[int(t)] for t in t_cpu],
            dtype=torch.long,
            device=device,
        )

        # ---- Extract head/tail embeddings from each view ----
        h_emb_orig = emb_orig[h_local_idx_orig]
        t_emb_orig = emb_orig[t_local_idx_orig]

        h_emb_aug1 = emb_aug1[h_local_idx_aug]
        t_emb_aug1 = emb_aug1[t_local_idx_aug]

        # ---- MLP logits ----
        logits_orig = mlp(h_emb_orig, t_emb_orig)
        logits_aug1 = mlp(h_emb_aug1, t_emb_aug1)

        # ---- Supervised loss on ORIGINAL view ----
        #
        # # ---- Consistency regularization between views ----
        loss_supervised = bce_loss_fn(logits_orig, y_batch)
        loss_cons1 = F.mse_loss(logits_orig, logits_aug1)
        l_con = 0.1 * loss_cons1 #0.1,0.2,0.05,0.4

        loss = loss_supervised + l_con
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


#below eval also return valid loss
# def eval_ddi(
#         gnn,
#         mlp,
#         kegg_triplets,
#         adjacency,
#         edges_per_node,
#         num_entities,
#         num_relations,
#         data_loader,
#         device,
#         bce_loss_fn,
# ):
#
#
#     """
#     Evaluate on a DDI split using node-centric subgraphs.
#     Returns a dict with acc, precision, recall, f1, auc, aupr, and 'loss'.
#     """
#     gnn.eval()
#     mlp.eval()
#
#     all_labels = []
#     all_probs = []
#
#     total_loss = 0.0
#     total_samples = 0
#
#
#
#     with torch.no_grad():
#         # for (h_batch, t_batch, y_batch) in data_loader:
#         for (h_batch, t_batch, y_batch) in tqdm(
#             data_loader,
#             desc="Eval",
#             total=len(data_loader),
#             leave=False,
#         ):
#             # Move to device
#             h_batch = h_batch.to(device)
#             t_batch = t_batch.to(device)
#             y_batch = y_batch.to(device).float()
#
#             # CPU numpy for sampling
#             h_cpu = h_batch.cpu().numpy()
#             t_cpu = t_batch.cpu().numpy()
#
#             # Build subgraph for this batch
#             subgraph_data, global_to_local = build_ddi_subgraph_for_batch_eval(  # build_ddi_subgraph_for_batch_eval
#                 batch_heads=h_cpu,
#                 batch_tails=t_cpu,
#                 adjacency=adjacency,
#                 kegg_triplets=kegg_triplets,
#                 edges_per_node=edges_per_node,
#                 num_entities=num_entities,
#                 num_relations=num_relations,
#             )
#
#             subgraph_data = subgraph_data.to(device)
#
#             # RGCN forward
#             emb = gnn(
#                 subgraph_data.entity,
#                 subgraph_data.edge_index,
#                 subgraph_data.edge_type,
#                 subgraph_data.edge_norm,
#             )
#
#             # Map to local indices
#             h_local_idx = torch.tensor(
#                 [global_to_local[int(h)] for h in h_cpu],
#                 dtype=torch.long,
#                 device=device,
#             )
#             t_local_idx = torch.tensor(
#                 [global_to_local[int(t)] for t in t_cpu],
#                 dtype=torch.long,
#                 device=device,
#             )
#
#             h_emb = emb[h_local_idx]
#             t_emb = emb[t_local_idx]
#
#             # MLP
#             logits = mlp(h_emb, t_emb)
#             probs = torch.sigmoid(logits)
#
#             # ----- accumulate loss -----
#             loss = bce_loss_fn(logits, y_batch)
#             batch_size = y_batch.size(0)
#             total_loss += loss.item() * batch_size
#             total_samples += batch_size
#             # ----------------------------
#
#             all_labels.append(y_batch.cpu().numpy())
#             all_probs.append(probs.detach().cpu().numpy())
#
#     # Concatenate and compute metrics
#     all_labels = np.concatenate(all_labels, axis=0)
#     all_probs = np.concatenate(all_probs, axis=0)
#
#     metrics = compute_classification_metrics(all_labels, all_probs)
#     avg_loss = total_loss / max(total_samples, 1)
#     metrics["loss"] = float(avg_loss)
#
#     return metrics

def eval_ddi_augmented(
        gnn,
        mlp,
        kegg_triplets,
        adjacency,
        edges_per_node,
        num_entities,
        num_relations,
        data_loader,
        device,
        bce_loss_fn,
        id2row2,
        vals2,
        id2row3,
        vals3,
        epoch,
        b,
):
    """
    Evaluate on a DDI split using the SAME two-view (orig+aug) subgraph construction
    used in training. Returns a dict with acc, precision, recall, f1, auc, aupr, and 'loss'.

    - gnn/mlp in eval mode
    - no optimizer / no backward
    - keeps orig+aug views and MSE consistency exactly like training
    - metrics computed from ORIGINAL view probs (sigmoid(logits_orig))
    """
    import numpy as np
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm

    gnn.eval()
    mlp.eval()

    all_labels = []
    all_probs = []

    #save
    all_heads = []
    all_tails = []

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for (h_batch, t_batch, y_batch) in tqdm(
            data_loader,
            desc="Eval",
            total=len(data_loader),
            leave=False,
        ):
            # Move ids + labels to device (same style as eval_ddi)
            h_batch = h_batch.to(device)
            t_batch = t_batch.to(device)
            y_batch = y_batch.to(device).float()

            # CPU numpy for sampling / subgraph construction
            h_cpu = h_batch.detach().cpu().numpy()
            t_cpu = t_batch.detach().cpu().numpy()

            # Build two views for this batch (same as training)
            data_orig, data_aug1, global_to_local_orig, global_to_local_aug, seed_local_orig, seed_local_aug = \
                build_ddi_subgraph_random_view1_view2_different_eval(
                    batch_heads=h_cpu,
                    batch_tails=t_cpu,
                    adjacency=adjacency,
                    kegg_triplets=kegg_triplets,
                    edges_per_node=edges_per_node,
                    num_entities=num_entities,
                    num_relations=num_relations,
                    id2row2=id2row2, vals2=vals2,
                    id2row3=id2row3, vals3=vals3,
                    epoch=epoch,
                )

            data_orig = data_orig.to(device)
            #data_aug1 = data_aug1.to(device)

            # RGCN forward on each view
            emb_orig = gnn(
                data_orig.entity,
                data_orig.edge_index,
                data_orig.edge_type,
                data_orig.edge_norm,
                debug_seed_nodes=seed_local_orig,
            )
            # emb_aug1 = gnn(
            #     data_aug1.entity,
            #     data_aug1.edge_index,
            #     data_aug1.edge_type,
            #     data_aug1.edge_norm,
            #     debug_seed_nodes=seed_local_aug,
            # )

            # Map batch entities to local indices in each view
            h_local_idx_orig = torch.tensor(
                [global_to_local_orig[int(h)] for h in h_cpu],
                dtype=torch.long,
                device=device,
            )
            t_local_idx_orig = torch.tensor(
                [global_to_local_orig[int(t)] for t in t_cpu],
                dtype=torch.long,
                device=device,
            )
            # h_local_idx_aug = torch.tensor(
            #     [global_to_local_aug[int(h)] for h in h_cpu],
            #     dtype=torch.long,
            #     device=device,
            # )
            # t_local_idx_aug = torch.tensor(
            #     [global_to_local_aug[int(t)] for t in t_cpu],
            #     dtype=torch.long,
            #     device=device,
            # )

            # Extract embeddings
            h_emb_orig = emb_orig[h_local_idx_orig]
            t_emb_orig = emb_orig[t_local_idx_orig]
            # h_emb_aug1 = emb_aug1[h_local_idx_aug]
            # t_emb_aug1 = emb_aug1[t_local_idx_aug]

            # MLP logits
            logits_orig = mlp(h_emb_orig, t_emb_orig)
            # logits_aug1 = mlp(h_emb_aug1, t_emb_aug1)

            # Probabilities for metrics (use ORIGINAL view)
            probs = torch.sigmoid(logits_orig)

            #save these below two lines save for target drug in excel
            all_heads.extend(h_cpu)
            all_tails.extend(t_cpu)

            # Loss = supervised (orig) + lambda * mse(orig, aug)
            loss_supervised = bce_loss_fn(logits_orig, y_batch)
            # loss_cons1 = F.mse_loss(logits_orig, logits_aug1)
            # l_con = 0.1 * loss_cons1
            # loss = loss_supervised + l_con

            # Accumulate loss weighted by batch size (same style as eval_ddi)
            batch_size = y_batch.size(0)
            total_loss += loss_supervised.item() * batch_size # use loss from loss = loss_supervised + l_con if use CR
            total_samples += batch_size

            all_labels.append(y_batch.detach().cpu().numpy())
            all_probs.append(probs.detach().cpu().numpy())

    # Concatenate and compute metrics (same as eval_ddi)
    all_labels = np.concatenate(all_labels, axis=0) if len(all_labels) else np.zeros((0,), dtype=np.float32)
    all_probs = np.concatenate(all_probs, axis=0)  if len(all_probs)  else np.zeros((0,), dtype=np.float32)

    if b:
        #save_full_test_set_with_probabilities(all_probs)

        save_full_test_set_with_probabilities(
            all_heads,
            all_tails,
            all_labels,
            all_probs
        )
        #OR SINGLE DRUGS
        # Below save excel call
        # save_positive_pairs_for_drug(
        #     drug_id=30194,  # change this to the drug you want
        #     heads=all_heads,
        #     tails=all_tails,
        #     labels=all_labels,
        #     probs=all_probs,
        #     output_file="30194_test_drug_positive_pairs.xlsx"
        # )
        # save_positive_pairs_for_drug(
        #     drug_id=12817,  # change this to the drug you want
        #     heads=all_heads,
        #     tails=all_tails,
        #     labels=all_labels,
        #     probs=all_probs,
        #     output_file="12817_test_drug_positive_pairs.xlsx"
        # )
        #
        # save_positive_pairs_for_drug(
        #     drug_id=13177,  # change this to the drug you want
        #     heads=all_heads,
        #     tails=all_tails,
        #     labels=all_labels,
        #     probs=all_probs,
        #     output_file="13177_test_drug_positive_pairs.xlsx"
        # )
        # save_positive_pairs_for_drug(
        #     drug_id=13534,  # change this to the drug you want
        #     heads=all_heads,
        #     tails=all_tails,
        #     labels=all_labels,
        #     probs=all_probs,
        #     output_file="13534_test_drug_positive_pairs.xlsx"
        # )



    metrics = compute_classification_metrics(all_labels, all_probs)
    avg_loss = total_loss / max(total_samples, 1)
    metrics["loss"] = float(avg_loss)

    return metrics







def main():


    parser = argparse.ArgumentParser(description="DDI prediction using pretrained RGCN + MLP")

    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory containing KEGG KG file and mapping files.")
    parser.add_argument("--kegg-file", type=str, default="ggsub-DRKG2.txt",
                        help="Single KEGG triples file (head relation tail).")
    parser.add_argument("--entities-file", type=str, default="ggentities2id.txt",
                        help="Entity mapping file (from original project).")
    parser.add_argument("--relations-file", type=str, default="ggrelations2id.txt",
                        help="Relation mapping file (from original project).")
    # parser.add_argument("--ddi-path", type=str, required=True,
    #                     help="DDI file with three columns: entity1 entity2 label.")

    # parser.add_argument("--ddi-path", type=str, required=True,
    #                     help="DDI file with three columns: entity1 entity2 label "
    #                          "(used only if CSV splits are not provided).")
    parser.add_argument("--ddi-train", type=str, default=None,
                        help="(Optional) CSV file with fixed TRAIN DDI pairs (entity1,entity2,label).")
    parser.add_argument("--ddi-valid", type=str, default=None,
                        help="(Optional) CSV file with fixed VALIDATION DDI pairs (entity1,entity2,label).")
    parser.add_argument("--ddi-test", type=str, default=None,
                        help="(Optional) CSV file with fixed TEST DDI pairs (entity1,entity2,label).")

    # parser.add_argument("--pretrained-path", type=str, required=True,
    #                     help="Path to pretrained RGCN checkpoint (e.g., 'augmented_RGCN moyuyudel.pth').")

    parser.add_argument("--gpu", type=int, default=-1,
                        help="GPU id to use; -1 for CPU.")
    parser.add_argument("--n-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64) #64 was used
    parser.add_argument("--n-bases", type=int, default=2) #2
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.01) #1e-2 = 0.01
    parser.add_argument("--mlp-hidden", type=int, default=200)
    parser.add_argument("--mlp-dropout", type=float, default=0.1)
    parser.add_argument("--weigh_d", type=float, default=0.0000001) #0.00005 (own before)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1) #(1:main)
    parser.add_argument("--num-hops", type=int, default=2,
                        help="Number of hops for node-centric subgraph sampling around DDI entities.")

    args = parser.parse_args()

    set_seed(args.seed)
    print("=====")
    print(args)
    print("====")

    print(f"DEBUG RNG ",
          "py", random.random(),
          "np", np.random.rand(),
          "torch", torch.rand(1).item())

    # Sanity check: if user starts using CSV splits, they must provide all three.
    if any([args.ddi_train, args.ddi_valid, args.ddi_test]):
        if not (args.ddi_train and args.ddi_valid and args.ddi_test):
            raise ValueError(
                "If you use fixed CSV splits, please provide all of "
                "--ddi-train, --ddi-valid and --ddi-test."
            )



    # -------------------------------------------------------------------------
    # 1. Device
    # -------------------------------------------------------------------------
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        print(f"Using GPU: cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
        print("Using CPU")


    # -------------------------------------------------------------------------
    # 2. Paths
    # -------------------------------------------------------------------------
    entities_path = os.path.join(args.data_dir, args.entities_file)
    relations_path = os.path.join(args.data_dir, args.relations_file)
    kegg_path = os.path.join(args.data_dir, args.kegg_file)

    # -------------------------------------------------------------------------
    # 3. Load mappings and KEGG KG
    # -------------------------------------------------------------------------
    print("Loading entity and relation mappings...")
    entity2id, relation2id = load_entity_relation_mappings(entities_path, relations_path)
    num_entities = max(entity2id.values()) + 1
    num_relations = max(relation2id.values()) + 1

    print(f"Loaded {len(entity2id)} entities, {len(relation2id)} relations.")
    print("Loading KEGG triples (single KG file)...")
    kegg_triplets = load_kegg_triples_single_file(kegg_path, entity2id, relation2id)
    print(f"Loaded {kegg_triplets.shape[0]} KEGG triples.")

    print("Building KEGG adjacency for node-centric sampling...")
    adjacency = build_kegg_adjacency(kegg_triplets, num_entities)
    #kg_pack = prepare_kg_gpu(kegg_triplets, device="cuda")



    # Build edges_per_node once for fast subgraph edge retrieval
    print("Building edges_per_node for fast subgraph extraction...")
    edges_per_node = [[] for _ in range(num_entities)]
    for e_idx, (h, r, t) in enumerate(kegg_triplets):
        h = int(h)
        t = int(t)
        edges_per_node[h].append(e_idx)
        edges_per_node[t].append(e_idx)
    print("Done building edges_per_node.")

    # -------------------------------------------------------------------------
    # 4. Load / prepare DDI dataset
    # -------------------------------------------------------------------------
    # Two modes:
    # (A) If --ddi-train/--ddi-valid/--ddi-test are given, read fixed CSV splits.
    # (B) Otherwise, read a single DDI file (args.ddi_path) and split randomly.
    if args.ddi_train is not None and args.ddi_valid is not None and args.ddi_test is not None:
        print("Loading DDI dataset from fixed CSV splits...")
        (h_train, t_train, y_train), (h_valid, t_valid, y_valid), (h_test, t_test, y_test) = \
            load_ddi_splits_from_csv(
                train_path=args.ddi_train,
                valid_path=args.ddi_valid,
                test_path=args.ddi_test,
                entity2id=entity2id,
                delimiter=",",  # your CSVs use comma as delimiter
            )
        print(f"DDI splits (fixed): train={len(y_train)}, valid={len(y_valid)}, test={len(y_test)}")


    # Create DataLoaders
    train_dataset = TensorDataset(
        torch.from_numpy(h_train),
        torch.from_numpy(t_train),
        torch.from_numpy(y_train),
    )
    valid_dataset = TensorDataset(
        torch.from_numpy(h_valid),
        torch.from_numpy(t_valid),
        torch.from_numpy(y_valid),
    )
    test_dataset = TensorDataset(
        torch.from_numpy(h_test),
        torch.from_numpy(t_test),
        torch.from_numpy(y_test),
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False)

    # -------------------------------------------------------------------------
    # 5. Build models and load pretrained RGCN
    # -------------------------------------------------------------------------
    print("Initializing RGCN and DDI MLP...")
    gnn = RGCN(
        num_entities=num_entities,
        num_relations=num_relations,
        num_bases=args.n_bases,
        dropout=args.dropout,
    )
    gnn.to(device)

    mlp = DDIMLP(
        embed_dim=100,          # must match RGCN embedding dim
        hidden_dim=args.mlp_hidden,
        dropout=args.mlp_dropout,
    )
    mlp.to(device)


    # =================================================

    # -------------------------------------------------------------------------
    # 6. Training setup
    # -------------------------------------------------------------------------
    bce_loss_fn = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        list(gnn.parameters()) + list(mlp.parameters()),
        lr=args.lr,
        weight_decay=args.weigh_d,
    )

    best_state = None

    best_valid_auc = 0.0  # start from 0, we want to maximize AUC
    epochs_no_improve = 0
    patience = 10  # keep same patience

    # print("Using device:", device, "| CUDA available:", torch.cuda.is_available())
    # exit()
    # neigh_2 = torch.load("neigh-2.pt")  # triadic 2-hop neighbors
    # neigh_3 = torch.load("neigh-3.pt")  # L3 3-hop neighbors

    # neigh_2 = torch.load("neigh-2.pt", weights_only=False)  # triadic 2-hop neighbors
    # neigh_3 = torch.load("neigh-3.pt", weights_only=False)  # l3 neighbors
    neigh_2 = torch.load("neigh-2.pt", map_location="cpu")
    neigh_3 = torch.load("neigh-3.pt", map_location="cpu")

    def build_fixedK(neigh_dict, K, pad=-1):
        keys = np.fromiter((int(k) for k in neigh_dict.keys()), dtype=np.int64)
        n = int(keys.size)
        if n == 0:
            id2row = np.full(1, -1, dtype=np.int32)
            vals = np.full((0, K), pad, dtype=np.int64)
            return id2row, vals

        max_id = int(keys.max())
        id2row = np.full(max_id + 1, -1, dtype=np.int32)
        id2row[keys] = np.arange(n, dtype=np.int32)

        vals = np.full((n, K), pad, dtype=np.int64)
        for row, u in enumerate(keys):
            lst = neigh_dict[int(u)]
            m = min(K, len(lst))
            if m:
                vals[row, :m] = np.asarray(lst[:m], dtype=np.int64)
        return id2row, vals

    # ---- fixed-K neighbor tables (CPU) ----
    K2 = 4
    K3 = 4
    id2row2_np, vals2_np = build_fixedK(neigh_2, K2)
    id2row3_np, vals3_np = build_fixedK(neigh_3, K3)

    id2row2 = torch.from_numpy(id2row2_np).to(torch.int32)   # CPU
    vals2   = torch.from_numpy(vals2_np).to(torch.int64)     # CPU
    id2row3 = torch.from_numpy(id2row3_np).to(torch.int32)   # CPU
    vals3   = torch.from_numpy(vals3_np).to(torch.int64)     # CPU

    for epoch in range(1, args.n_epochs + 1):
        train_loss = train_epoch_ddi_augmented( #call to train_epoch_ddi or train_epoch_ddi_augmented
            gnn=gnn,
            mlp=mlp,
            kegg_triplets=kegg_triplets,
            adjacency=adjacency,
            edges_per_node=edges_per_node,  # NEW
            num_entities=num_entities,
            num_relations=num_relations,
            data_loader=train_loader,
            device=device,
            bce_loss_fn=bce_loss_fn,
            optimizer=optimizer,
            epoch=epoch,
            id2row2=id2row2, vals2=vals2,
            id2row3=id2row3, vals3=vals3,
            #kg_pack=kg_pack,
        )
        # print(
        #     f"Epoch {epoch}: "
        #     f"gamma1={torch.sigmoid(gnn.conv1.lambda_aug).item():.4f}, "
        #     f"gamma2={torch.sigmoid(gnn.conv2.lambda_aug).item():.4f}"
        # )

        # ---- ADD THIS BLOCK HERE ----

        # -----------------------------

        #bwlow for matrics and valid loss
        # valid_metrics = eval_ddi(  # use eval without aug, simple eval
        #     gnn=gnn,
        #     mlp=mlp,
        #     kegg_triplets=kegg_triplets,
        #     adjacency=adjacency,
        #     edges_per_node=edges_per_node,
        #     num_entities=num_entities,
        #     num_relations=num_relations,
        #     data_loader=valid_loader,
        #     device=device,
        #     bce_loss_fn=bce_loss_fn,
        # )


        #OR

        #DEBUG
        # import itertools as it
        #
        # def first2_repeat2(loader):
        #     first_two = list(it.islice(loader, 2))  # take only first 2 batches
        #     for b in first_two:
        #         yield b
        #         yield b
        #
        # debug_val_loader = list(first2_repeat2(valid_loader))  # now length = 4

        valid_metrics = eval_ddi_augmented(  # call to train_epoch_ddi or train_epoch_ddi_augmented
            gnn=gnn,
            mlp=mlp,
            kegg_triplets=kegg_triplets,
            adjacency=adjacency,
            edges_per_node=edges_per_node,  # NEW
            num_entities=num_entities,
            num_relations=num_relations,
            data_loader=valid_loader,
            device=device,
            bce_loss_fn=bce_loss_fn,
            #optimizer=optimizer,
            id2row2=id2row2, vals2=vals2,
            id2row3=id2row3, vals3=vals3,
            epoch=epoch,
            b=False,
            # kg_pack=kg_pack,
        )

        print(
            f"[Epoch {epoch:03d}] "
            f"Train loss: {train_loss:.4f} | "
            f"valid loss: {valid_metrics['loss']:.4f}, "
            f"Valid: acc={valid_metrics['acc']:.4f}, "
            f"precision={valid_metrics['precision']:.4f}, "
            f"recall={valid_metrics['recall']:.4f}, "
            f"f1={valid_metrics['f1']:.4f}, "
            f"auc={valid_metrics['auc']:.4f}, "
            f"aupr={valid_metrics['aupr']:.4f}"
        )




        #Bwlow us based on AUC stopping criteria

        # ---- Early stopping logic based on validation AUC ----
        current_auc = valid_metrics["auc"]

        # TRUNCATE to 3 decimals (NO rounding)
        #current_auc = int(current_auc * 1000) / 1000.0

        if current_auc > best_valid_auc:
            best_valid_auc = current_auc
            epochs_no_improve = 0
            best_state = {
                "gnn_state": gnn.state_dict(),
                "mlp_state": mlp.state_dict(),
                "epoch": epoch,
                "valid_metrics": valid_metrics,
            }
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(
                f"Early stopping triggered at epoch {epoch} "
                f"(no validation AUC improvement for {epochs_no_improve} epochs)."
            )
            break


    # -------------------------------------------------------------------------
    # 8. Test evaluation with best model (if any)
    # -------------------------------------------------------------------------
    if best_state is not None:
        print(
            f"Loading best model from epoch {best_state['epoch']} "
            f"(best val AUC={best_valid_auc:.4f})"
        )
        gnn.load_state_dict(best_state["gnn_state"])
        mlp.load_state_dict(best_state["mlp_state"])

    print("Evaluating on test set...")
    # test_metrics = eval_ddi(
    #     gnn=gnn,
    #     mlp=mlp,
    #     kegg_triplets=kegg_triplets,
    #     adjacency=adjacency,
    #     edges_per_node=edges_per_node,
    #     num_entities=num_entities,
    #     num_relations=num_relations,
    #     data_loader=test_loader,
    #     device=device,
    #     bce_loss_fn=bce_loss_fn,  # ← ADD THIS
    # )

    #OR
    epoch = 100
    test_metrics = eval_ddi_augmented(  # call to train_epoch_ddi or train_epoch_ddi_augmented
        gnn=gnn,
        mlp=mlp,
        kegg_triplets=kegg_triplets,
        adjacency=adjacency,
        edges_per_node=edges_per_node,  # NEW
        num_entities=num_entities,
        num_relations=num_relations,
        data_loader=test_loader,
        device=device,
        bce_loss_fn=bce_loss_fn,
        # optimizer=optimizer,
        id2row2=id2row2, vals2=vals2,
        id2row3=id2row3, vals3=vals3,
        epoch=epoch,
        b=True,
        # kg_pack=kg_pack,
    )

    print(
        f"Test: acc={test_metrics['acc']:.4f}, "
        f"precision={test_metrics['precision']:.4f}, "
        f"recall={test_metrics['recall']:.4f}, "
        f"f1={test_metrics['f1']:.4f}, "
        f"auc={test_metrics['auc']:.4f}, "
        f"aupr={test_metrics['aupr']:.4f}"
    )

if __name__ == "__main__":

    main()
