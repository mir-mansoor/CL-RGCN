import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
from utils import uniform


class RGCN(nn.Module):
    def __init__(self, num_entities, num_relations, num_bases, dropout):
        super(RGCN, self).__init__()

        # ORIGINAL
        self.entity_embedding = nn.Embedding(num_entities, 100)
        self.relation_embedding = nn.Parameter(torch.Tensor(num_relations, 100))
        nn.init.xavier_uniform_(self.relation_embedding, gain=nn.init.calculate_gain('relu'))

        # AUG IDS (must match builder)
        self.triadic_rel_id = 42
        self.l3_rel_id = 43

        # ORIGINAL STRUCTURE: 2 conv layers, same dims
        self.conv1 = RGCNConv(
            100, 100, num_relations * 2,
            num_bases=num_bases,
            use_aug_weight=True, #undo
        )
        self.conv2 = RGCNConv(
            100, 100, num_relations * 2,
            num_bases=num_bases,
            use_aug_weight=True, #undo   # conv2 has its own GAT params
        )

        self.dropout_ratio = dropout
        self.attn_temp = 1.0


        # ----------------------------
        # STANDARD GAT DROPOUTS
        # ----------------------------
        self.gat_feat_drop = 0.1 #0.1  # feature dropout before W(.)
        self.gat_attn_drop = 0.0   # attention dropout after softmax

    @torch.no_grad()
    def _compute_aug_mask(self, edge_type, base_R):
        base_rel = edge_type % base_R
        return (base_rel == self.triadic_rel_id) | (base_rel == self.l3_rel_id)



    def _gat_alpha_and_msgproj_for_aug_edges(self, conv, x, edge_index, edge_is_aug, edge_type, debug_seed_nodes=None):


        bool = False
        """
        Returns:
          alpha_aug: [E_aug] attention weights for augmented edges (softmax over dst)
          Wx_src:    [E_aug, F] projected source features W x_j for augmented edges (message content)

        NOTE: This uses the provided conv's (layer-specific) gat_W and gat_a.
        """
        src_aug = edge_index[0, edge_is_aug]
        dst_aug = edge_index[1, edge_is_aug]
        E_aug = int(src_aug.numel())
        if E_aug == 0:
            return None, None

        # ---- Compute W(x) ONLY for nodes touched by augmented edges ----
        all_nodes = torch.cat([src_aug, dst_aug], dim=0)  # [2*E_aug]
        uniq_nodes, inv = torch.unique(all_nodes, return_inverse=True)

        # ---- STANDARD GAT: feature dropout BEFORE projection ----
        x_drop = F.dropout(x, p=self.gat_feat_drop, training=self.training)

        Wx_uniq = conv.gat_W(x_drop[uniq_nodes])  # [U, F]
        inv_src = inv[:E_aug]
        inv_dst = inv[E_aug:]

        Wx_src = Wx_uniq[inv_src]  # [E_aug, F]
        Wx_dst = Wx_uniq[inv_dst]  # [E_aug, F]

        # ---- No concat: split a into (a_dst, a_src) ----
        a = conv.gat_a  # [2F]
        Fdim = Wx_src.size(-1)
        a_dst = a[:Fdim]
        a_src = a[Fdim:]

        # e_ij = LeakyReLU( a_dst^T Wx_i + a_src^T Wx_j )
        e = (Wx_dst * a_dst).sum(dim=-1) + (Wx_src * a_src).sum(dim=-1)
        e = F.leaky_relu(e, negative_slope=0.2)

        # alpha over incoming edges per dst node (pure GAT normalization)
        alpha = softmax(e / self.attn_temp,dst_aug)  # [E_aug] #this is original GAT attention as per paper (it gives relative scores
        # not rejects. I needs more than one to give relative score 1-0 each time
        # OR
        # alpha = torch.sigmoid(e)  # [E_aug]             # This is if we want to reject the aug edges completely if not beneficial

        # ---- STANDARD GAT: attention dropout AFTER softmax ----
        alpha = F.dropout(alpha, p=self.gat_attn_drop, training=self.training)


        #
        # # ======================= DEBUG: AUGMENTED EDGE COUNTS PER SEED =======================
        if bool:
            if debug_seed_nodes is not None:
                seed_nodes = torch.as_tensor(
                    debug_seed_nodes, device=src_aug.device, dtype=src_aug.dtype
                )
                seed_nodes = torch.unique(seed_nodes)

                print("\n[GAT DEBUG] Augmented-edge summary per seed (LOCAL ids)")
                print("  seed |  IN | OUT | SELF | TOTAL")
                print("  --------------------------------")

                for d in seed_nodes.tolist():
                    in_cnt = int(((dst_aug == d) & (src_aug != d)).sum().item())
                    out_cnt = int(((src_aug == d) & (dst_aug != d)).sum().item())
                    self_cnt = int(((src_aug == d) & (dst_aug == d)).sum().item())
                    total = in_cnt + out_cnt + self_cnt

                    if total > 0:  # print only seeds that actually have augmented edges
                        print(f"  {d:5d} | {in_cnt:3d} | {out_cnt:3d} | {self_cnt:4d} | {total:5d}")

                print("  --------------------------------")




        # HARD-CODE the LOCAL seed you want to inspect
        DEBUG_SEED_LOCAL = 1342
        if bool:
            if debug_seed_nodes is not None:
                seed_nodes = torch.as_tensor(
                    debug_seed_nodes, device=src_aug.device
                )
                seed_nodes = torch.unique(seed_nodes)

                # which seeds appear in augmented edges
                has_edge = torch.isin(seed_nodes, src_aug) | torch.isin(seed_nodes, dst_aug)

                seeds_with_edges = seed_nodes[has_edge]

                print(
                    "Seed nodes with ≥1 augmented edge:",
                    " ".join(map(str, seeds_with_edges.tolist()))
                )

            # int(debug_seed_nodes[0])  # or set explicitly, e.g. = 1919
            d = DEBUG_SEED_LOCAL

            mask = (src_aug == d) | (dst_aug == d)
            idx = torch.where(mask)[0]

            if idx.numel() == 0:
                print(f"[GAT DEBUG] DRUG={d} | no augmented edges")
            else:
                in_cnt = int(((dst_aug == d) & (src_aug != d)).sum().item())
                out_cnt = int(((src_aug == d) & (dst_aug != d)).sum().item())
                self_cnt = int(((src_aug == d) & (dst_aug == d)).sum().item())

                print(
                    f"\n[GAT DEBUG] DRUG={d} | AUG_EDGES={int(idx.numel())} | "
                    f"IN={in_cnt} | OUT={out_cnt} | SELF={self_cnt}"
                )

                for i in idx.tolist():
                    s = int(src_aug[i])
                    t = int(dst_aug[i])
                    r = int(edge_type[edge_is_aug][i])
                    a = float(alpha[i])

                    direction = "SELF" if s == t else ("OUT" if s == d else "IN")

                    print(
                        f"  edge_idx={i:4d} | {direction:4s} | "
                        f"{s} -[{r}]-> {t} | alpha={a:.6f}"
                    )


        # ================================================================================

        #OR

        if debug_seed_nodes is not None:
            K = 2  #Ather
            src = src_aug.long().view(-1)
            dst = dst_aug.long().view(-1)
            alpha = alpha.view(-1).contiguous()

            E = alpha.numel()
            if src.numel() != E or dst.numel() != E:
                return alpha, Wx_src  # mismatch upstream; don't crash

            seeds = torch.as_tensor(debug_seed_nodes, device=src.device, dtype=torch.long).unique()
            if seeds.numel() == 0:
                return alpha, Wx_src
            seeds, _ = torch.sort(seeds)
            S = int(seeds.numel())

            # membership + seed-position (0..S-1), safe on CUDA
            idx = torch.searchsorted(seeds, dst)
            idxc = idx.clamp_max(S - 1)
            is_seed_dst = (idx < S) & (seeds[idxc] == dst)
            pos_dst = idxc

            idx = torch.searchsorted(seeds, src)
            idxc = idx.clamp_max(S - 1)
            is_seed_src = (idx < S) & (seeds[idxc] == src)
            pos_src = idxc

            alpha_new = alpha.new_zeros(alpha.size())

            # self-loop on seed dst: set to 0
            self_m = is_seed_dst & (src == dst)
            alpha_new[self_m] = 0.0

            def keep_topk(edge_m, group_pos):
                eidx = torch.nonzero(edge_m, as_tuple=True)[0]
                if eidx.numel() == 0:
                    return

                g = group_pos[eidx]  # [M] in 0..S-1
                a = alpha[eidx]  # [M]

                # stable sort by score desc, then stable sort by group asc
                p_score = torch.argsort(a, descending=True)
                g2 = g[p_score]
                p = p_score[torch.argsort(g2, stable=True)]

                eidx = eidx[p]
                g = g[p]

                # rank within each group (0,1,2,..) and keep rank < K
                pos = torch.arange(g.numel(), device=g.device)
                is_new = torch.ones_like(g, dtype=torch.bool)
                is_new[1:] = g[1:] != g[:-1]
                start = torch.where(is_new, pos, pos.new_zeros(()).expand_as(pos))
                start = torch.cummax(start, dim=0).values
                rank = pos - start

                chosen = eidx[rank < K]
                alpha_new[chosen] = alpha[chosen]

            # IN: dst is seed, exclude self
            keep_topk(is_seed_dst & (src != dst), pos_dst)

            # OUT: src is seed, exclude self
            #keep_topk(is_seed_src & (dst != src), pos_src)

            alpha = alpha_new

            # renormalize per-dst (safe w.r.t. node count)
            num_nodes = int(x.size(0))
            valid = (dst >= 0) & (dst < num_nodes)
            den = alpha.new_zeros(num_nodes)
            den.scatter_add_(0, dst[valid], alpha[valid])
            eps = torch.finfo(alpha.dtype).eps
            alpha_valid = alpha[valid] / (den[dst[valid]] + eps)
            alpha = alpha.new_zeros(alpha.size())
            alpha[valid] = alpha_valid

            # --------------------------------------------------------

        # ================================================================================
        # print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        #
        if bool:
            d = DEBUG_SEED_LOCAL

            mask = (src_aug == d) | (dst_aug == d)
            idx = torch.where(mask)[0]

            if idx.numel() == 0:
                print(f"[GAT DEBUG] DRUG={d} | no augmented edges")
            else:
                in_cnt = int(((dst_aug == d) & (src_aug != d)).sum().item())
                out_cnt = int(((src_aug == d) & (dst_aug != d)).sum().item())
                self_cnt = int(((src_aug == d) & (dst_aug == d)).sum().item())

                print(
                    f"\n[GAT DEBUG] DRUG={d} | AUG_EDGES={int(idx.numel())} | "
                    f"IN={in_cnt} | OUT={out_cnt} | SELF={self_cnt}"
                )

                for i in idx.tolist():
                    s = int(src_aug[i])
                    t = int(dst_aug[i])
                    r = int(edge_type[edge_is_aug][i])
                    a = float(alpha[i])

                    direction = "SELF" if s == t else ("OUT" if s == d else "IN")

                    print(
                        f"  edge_idx={i:4d} | {direction:4s} | "
                        f"{s} -[{r}]-> {t} | alpha={a:.6f}"
                    )
        if bool:
            exit()

        # return hard-gated alpha
        return alpha, Wx_src




    def forward(self, entity, edge_index, edge_type, edge_norm, debug_seed_nodes=None):
        x0 = self.entity_embedding(entity)

        # ----------------------------
        # Detect augmented edges ONCE
        # ----------------------------
        base_R = self.conv1.num_relations // 2
        mask_aug = self._compute_aug_mask(edge_type, base_R)
        edge_is_aug = mask_aug if mask_aug.any() else None

        # ---------------------------------------------------------
        # STANDARD GAT: add self-loops to the *augmented attention* graph
        # Only when augmented edges exist (and only for nodes that touch aug edges)
        # ---------------------------------------------------------
        # if edge_is_aug is not None: Use if include selfoop aug edge or else use below
        if False and edge_is_aug is not None:
            aug_nodes = torch.unique(edge_index[:, edge_is_aug].reshape(-1))

            # i -> i self-loops
            loop_edge_index = torch.stack([aug_nodes, aug_nodes], dim=0)  # [2, N_loop]

            # Mark self-loops as augmented by using an augmented base relation id
            loop_edge_type = torch.full(
                (aug_nodes.numel(),),
                int(self.triadic_rel_id),
                device=edge_type.device,
                dtype=edge_type.dtype
            )

            # edge_norm for loops can be 1.0 (and will be canceled anyway for aug edges)
            if edge_norm is None:
                loop_edge_norm = None
            else:
                loop_edge_norm = torch.ones(
                    (aug_nodes.numel(),),
                    device=edge_norm.device,
                    dtype=edge_norm.dtype
                )

            # append loops to the existing edge list
            edge_index = torch.cat([edge_index, loop_edge_index], dim=1)
            edge_type = torch.cat([edge_type, loop_edge_type], dim=0)
            if edge_norm is not None:
                edge_norm = torch.cat([edge_norm, loop_edge_norm], dim=0)

            # extend aug mask to include appended loop edges
            edge_is_aug = torch.cat(
                [edge_is_aug,
                 torch.ones(aug_nodes.numel(), device=edge_is_aug.device, dtype=torch.bool)],
                dim=0
            )

        # =========================================================
        # conv1: GAT ONLY on augmented edges (fast)
        # =========================================================
        alpha1_aug = None
        Wx1_src_aug = None
        if edge_is_aug is not None:
            alpha1_aug, Wx1_src_aug = self._gat_alpha_and_msgproj_for_aug_edges(
                self.conv1, x0, edge_index, edge_is_aug, edge_type, debug_seed_nodes
            ) #undo

        x = F.relu(self.conv1(
            x0, edge_index, edge_type, edge_norm,
            edge_is_aug=edge_is_aug,
            alpha_aug=alpha1_aug,         # [E_aug] only
            Wx_src_aug=Wx1_src_aug,       # [E_aug, F] only
        ))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)

        # =========================================================
        # conv2: recompute GAT ONLY on augmented edges (fast)
        # =========================================================
        alpha2_aug = None
        Wx2_src_aug = None
        if edge_is_aug is not None:
            alpha2_aug, Wx2_src_aug = self._gat_alpha_and_msgproj_for_aug_edges(
                self.conv2, x, edge_index, edge_is_aug, edge_type,debug_seed_nodes
            ) #undo

        x = self.conv2(
            x, edge_index, edge_type, edge_norm,
            edge_is_aug=edge_is_aug,
            alpha_aug=alpha2_aug,         # [E_aug] only
            Wx_src_aug=Wx2_src_aug,       # [E_aug, F] only
        )
        return x


class RGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_relations, num_bases,
                 root_weight=True, bias=True, use_aug_weight=False, **kwargs):
        super(RGCNConv, self).__init__(aggr='sum', **kwargs)

        # basis → shared transformation matrices
        # att → relation - specific mixing weights
        # root → self - feature transform
        # bias → output shift

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases

        self.basis = nn.Parameter(torch.Tensor(num_bases, in_channels, out_channels))
        self.att = nn.Parameter(torch.Tensor(num_relations, num_bases))

        if root_weight:
            self.root = nn.Parameter(torch.Tensor(in_channels, out_channels))
        else:
            self.register_parameter('root', None)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        # GAT params exist only if use_aug_weight=True
        self.use_aug_weight = use_aug_weight
        if use_aug_weight:
            # project to out_channels (works even if in!=out)
            self.gat_W = nn.Linear(in_channels, out_channels, bias=False) #main projection layer



            self.gat_a = nn.Parameter(torch.empty(2 * out_channels))
            nn.init.xavier_uniform_(self.gat_W.weight)
            nn.init.xavier_uniform_(self.gat_a.view(1, -1))

            # ✅ ADD THIS:
            self.lambda_aug = nn.Parameter(torch.tensor(1.0)) #0000 #-6.90 Ather
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.basis)
        nn.init.xavier_uniform_(self.att)
        if self.root is not None:
            nn.init.xavier_uniform_(self.root)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, edge_type, edge_norm=None, size=None,
                edge_is_aug=None, alpha_aug=None, Wx_src_aug=None):
        """
        alpha_aug:   [E_aug] attention for augmented edges only
        Wx_src_aug:  [E_aug, F_out] projected source features (W x_j) for augmented edges only
        """
        return self.propagate(
            edge_index,
            size=size,
            x=x,
            edge_type=edge_type,
            edge_norm=edge_norm,
            edge_is_aug=edge_is_aug,
            alpha_aug=alpha_aug,
            Wx_src_aug=Wx_src_aug,
        )

    def message(self, x_j, edge_index_j, edge_type, edge_norm,
                edge_is_aug, alpha_aug, Wx_src_aug):
        #note x_j is the features of source nodes (done PyG internally)
        #edge_index_j this is the index values of source nodes (j is source)

        # ----- ORIGINAL R-GCN transform -----
        w = torch.matmul(self.att, self.basis.view(self.num_bases, -1))

        if x_j is None:
            w = w.view(-1, self.out_channels)
            index = edge_type * self.in_channels + edge_index_j
            out = torch.index_select(w, 0, index)
        else:
            w = w.view(self.num_relations, self.in_channels, self.out_channels)
            w = torch.index_select(w, 0, edge_type)
            out = torch.bmm(x_j.unsqueeze(1), w).squeeze(-2)

            # ----- GAT ONLY on augmented edges: message = W x_j -----
            if edge_is_aug is not None and edge_is_aug.any(): #undo
                if Wx_src_aug is None:
                    raise RuntimeError("Wx_src_aug must be provided when edge_is_aug is set.")
                out[edge_is_aug] = Wx_src_aug  # [E_aug, out_channels]

        # ----- ORIGINAL edge_norm (then cancel on aug edges like your code) -----
        if edge_norm is not None:
            norm = edge_norm.view(-1, 1)
            out = out * norm
            # below remove to carry norm for edges
            if edge_is_aug is not None and edge_is_aug.any(): #undo
                out[edge_is_aug] = out[edge_is_aug] / norm[edge_is_aug]

        # ----- Apply attention ONLY on augmented edges ----- #remove if want to use norm
        # if edge_is_aug is not None and edge_is_aug.any():
        #     if alpha_aug is None:
        #         raise RuntimeError("alpha_aug must be provided when edge_is_aug is set.")
        #     out[edge_is_aug] = out[edge_is_aug] * alpha_aug.view(-1, 1)

        #OR Above is no lembda global or below is lembda blobal
        if edge_is_aug is not None and edge_is_aug.any(): #undo
            if alpha_aug is None:
                raise RuntimeError("alpha_aug must be provided when edge_is_aug is set.")
            gamma = torch.sigmoid(self.lambda_aug)
            out[edge_is_aug] = out[edge_is_aug] * alpha_aug.view(-1, 1) * gamma

        return out

    def update(self, aggr_out, x):
        if self.root is not None:
            if x is None:
                out = aggr_out + self.root
            else:
                out = aggr_out + torch.matmul(x, self.root)
        else:
            out = aggr_out

        if self.bias is not None:
            out = out + self.bias
        return out

    def __repr__(self):
        return '{}({}, {}, num_relations={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.num_relations
        )


class DDIMLP(nn.Module):
    def __init__(self, embed_dim=100, hidden_dim=128, dropout=0.3): #hidden_dim=128 always for 100 embeddings
        super().__init__()
        self.fc1 = nn.Linear(embed_dim * 2, hidden_dim)
        self.bn = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, h, t):
        x = torch.cat([h, t], dim=-1)
        x = self.fc1(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        return self.fc_out(x).squeeze(-1)



