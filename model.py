import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LorentzLoss(nn.Module):
    def __init__(self, temperature=0.07, entailment_lambda=0.2, K=0.1):
        super().__init__()
        self.temperature = temperature
        self.entailment_lambda = entailment_lambda
        self.K = K  # Constraint value for the entailment cone's proximity to the origin

    def lorentz_inner_product(self, u, v):
        return torch.sum(u[..., :-1] * v[..., :-1], dim=-1) - u[..., -1] * v[..., -1]

    def lorentz_distance(self, u, v, c):
        k = 1.0 / c
        sqrt_k = torch.sqrt(k)
        inner_prod = self.lorentz_inner_product(u, v)
        acosh_input = torch.clamp(-inner_prod * c, min=1.0 + 1e-7)
        dist = sqrt_k * torch.acosh(acosh_input)
        return dist

    def contrastive_loss(self, txt_h, pcd_h, pos_mask, c):
        # Calculate the similarity matrix, using positive Lorentz inner product
        # (or negative Lorentz distance)
        sim_matrix = self.lorentz_inner_product(txt_h.unsqueeze(1), pcd_h.unsqueeze(0))
        # sim_matrix = -self.lorentz_distance(txt_h.unsqueeze(1), pcd_h.unsqueeze(0), c)
        logits = sim_matrix / self.temperature

        # Text-to-PointCloud loss
        numerator_i = torch.logsumexp(logits.masked_fill(~pos_mask, -float('inf')), dim=1)
        denominator_i = torch.logsumexp(logits, dim=1)
        loss_i = (denominator_i - numerator_i).mean()

        # PointCloud-to-Text loss
        logits_t = logits.T
        pos_mask_t = pos_mask.T
        numerator_t = torch.logsumexp(logits_t.masked_fill(~pos_mask_t, -float('inf')), dim=1)
        denominator_t = torch.logsumexp(logits_t, dim=1)
        loss_t = (denominator_t - numerator_t).mean()

        return (loss_i + loss_t) / 2.0

    def entailment_loss(self, txt_h, pcd_h, pos_mask, c):
        """
        Entailment loss, calculated only for positive pairs
        :param txt_h: Text global features (B, D+1)
        :param pcd_h: Point cloud global features (B, D+1)
        :param pos_mask: Mask matrix for positive pairs (B, B)
        """
        B = txt_h.shape[0]
        # 1. Calculate the half-aperture of all text entailment cones, aper(x)
        txt_space = txt_h[..., :-1]
        txt_space_norm = txt_space.norm(dim=-1).clamp(min=1e-8)  # (B)
        aper_x_input = torch.clamp((2 * self.K) / (torch.sqrt(c) * txt_space_norm), -1.0 + 1e-7, 1.0 - 1e-7)
        aper_x = torch.asin(aper_x_input)  # (B)

        # 2. Calculate the exterior angle ext(x, y) between all B x B pairs
        txt_h_b = txt_h.unsqueeze(1).expand(-1, B, -1)  # (B, B, D+1)
        pcd_h_b = pcd_h.unsqueeze(0).expand(B, -1, -1)  # (B, B, D+1)

        txt_time_b = txt_h_b[..., -1]
        pcd_time_b = pcd_h_b[..., -1]
        txt_space_norm_b = txt_space_norm.unsqueeze(1).expand(-1, B)  # (B, B)

        # Calculate inner product for all pairs
        inner_prod_xy = self.lorentz_inner_product(txt_h_b, pcd_h_b)  # (B, B)

        numerator = pcd_time_b + txt_time_b * c * inner_prod_xy
        sqrt_term_input = torch.clamp((c * inner_prod_xy) ** 2 - 1, min=1e-8)
        denominator = txt_space_norm_b * torch.sqrt(sqrt_term_input)
        ext_xy_input = torch.clamp(numerator / denominator.clamp(min=1e-8), -1.0 + 1e-7, 1.0 - 1e-7)
        ext_xy_matrix = torch.acos(ext_xy_input)  # (B, B)

        # 3. Calculate the entailment loss for all pairs
        potential_loss = F.relu(ext_xy_matrix - aper_x.unsqueeze(1))  # (B, B)

        # 4. Sum and average the loss only for positive pairs
        masked_loss = potential_loss * pos_mask.float()

        num_pos_pairs = pos_mask.sum().clamp(min=1.0)

        final_loss = masked_loss.sum() / num_pos_pairs

        return final_loss

    def forward(self, txt_global_h, pcd_global_h, pc_indices, c):
        # 1. Construct the positive sample mask based on pc_indices
        pos_mask = (pc_indices.unsqueeze(1) == pc_indices.unsqueeze(0))

        # 2. Calculate the multi-positive contrastive loss
        loss_cont = self.contrastive_loss(txt_global_h, pcd_global_h, pos_mask, c)

        # 3. Calculate the entailment loss, only for positive pairs
        loss_entail = self.entailment_loss(txt_global_h, pcd_global_h, pos_mask, c)

        total_loss = loss_cont + self.entailment_lambda * loss_entail

        return total_loss


class SelfAttentionEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(SelfAttentionEncoder, self).__init__()
        self.embed_dim = embed_dim
        if input_dim != embed_dim:
            self.input_proj = nn.Linear(input_dim, embed_dim)
        else:
            self.input_proj = nn.Identity()
        # self.input_proj = nn.Linear(input_dim, embed_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,  # Data format is (batch, seq, feature)
            norm_first=True  # Use Pre-LN (LayerNorm) structure
        )

    def forward(self, src):
        projected_src = self.input_proj(src)
        output = self.encoder_layer(projected_src)

        return output


class LorentzContributionAggregator(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def project_to_hyperboloid(self, x_euc: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Project Euclidean vectors to the Lorentz hyperboloid."""
        k = 1.0 / c
        x_space = x_euc
        # xtime = sqrt(1/c + ||xspace||^2)
        x_time = torch.sqrt(k + torch.sum(x_space ** 2, dim=-1, keepdim=True))
        return torch.cat([x_space, x_time], dim=-1)

    def lorentz_distance(self, u: torch.Tensor, v: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Calculate the Lorentz distance."""
        k_inv = c
        sqrt_k = torch.sqrt(1.0 / c)
        inner_prod = self.lorentz_inner_product(u, v)
        acosh_input = torch.clamp(-inner_prod * k_inv, min=1.0 + 1e-7)
        dist = sqrt_k * torch.acosh(acosh_input)
        return dist

    def lorentz_inner_product(self, u, v):
        """Calculate the Lorentz inner product."""
        uv_space = torch.sum(u[..., :-1] * v[..., :-1], dim=-1, keepdim=True)
        uv_time = u[..., -1:] * v[..., -1:]
        return uv_space - uv_time

    def forward(self, local_features_euc: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # local_features_euc: (B, L, D)
        # 1. Calculate an initial anchor in Euclidean space
        initial_topic_euc = local_features_euc.mean(dim=1, keepdim=True)  # (B, 1, D)

        # 2. Project all local tokens and the initial anchor to hyperbolic space
        leaf_nodes_h = self.project_to_hyperboloid(local_features_euc, c)  # (B, L, D+1)
        topic_h = self.project_to_hyperboloid(initial_topic_euc, c)  # (B, 1, D+1)

        # 3. Calculate Lorentz distance between each leaf node and the anchor to get contribution scores
        dists = self.lorentz_distance(leaf_nodes_h, topic_h, c)  # (B, L, 1)
        attention_logits = -dists.squeeze(-1) / self.temperature  # (B, L)

        # 4. Calculate contribution weights
        attention_weights = F.softmax(attention_logits, dim=1).unsqueeze(-1)  # (B, L, 1)

        # 5. Perform a weighted sum on the original Euclidean tokens using the weights
        final_global_euc = (attention_weights * local_features_euc).sum(dim=1)  # (B, D)

        # 6. Project the final aggregated global feature to hyperbolic space to form the true root node
        final_global_h = self.project_to_hyperboloid(final_global_euc, c)  # (B, D+1)

        return final_global_h
        # return topic_h.squeeze(1)  # W/o contribution-aware aggregation


class RetrievalModel(nn.Module):
    def __init__(self, d_model: int, d_txt: int, d_pcd: int, n_head: int, dropout: float = 0.1,
                 num_layers: int = 6, hyperbolic_c: float = 1.0,
                 loss_temp: float = 0.07, loss_lambda: float = 0.1):
        super(RetrievalModel, self).__init__()
        # 1. Context encoders in Euclidean space
        self.text_encoder = nn.ModuleList(
            [SelfAttentionEncoder(d_txt if i == 0 else d_model, d_model, n_head,
                                  ff_dim=d_model * 4, dropout=dropout) for i in range(num_layers)]
        )
        self.pcd_encoder = nn.ModuleList(
            [SelfAttentionEncoder(d_pcd if i == 0 else d_model, d_model, n_head,
                                  ff_dim=d_model * 4, dropout=dropout) for i in range(num_layers)]
        )

        # 2. Contribution-aware feature aggregators
        self.text_aggregator = LorentzContributionAggregator()
        self.pcd_aggregator = LorentzContributionAggregator()

        # 3. Core learnable parameters
        # 3.1 Learnable curvature c
        # Learn log(c) to ensure c is positive, initialized as log(1.0)=0
        self.log_c = nn.Parameter(torch.tensor(math.log(hyperbolic_c)))

        # 3.2 Learnable scaling factor α to prevent numerical overflow
        # Initialize α = 1/sqrt(d_model) and learn log(α)
        self.log_alpha_txt = nn.Parameter(torch.tensor(math.log(1.0 / math.sqrt(d_model))))
        self.log_alpha_pcd = nn.Parameter(torch.tensor(math.log(1.0 / math.sqrt(d_model))))

        # 4. Loss function module
        self.loss_fn = LorentzLoss(temperature=loss_temp, entailment_lambda=loss_lambda)

    def forward(self, txt: torch.Tensor, point_cloud: torch.Tensor, pc_indices: torch.Tensor = None):
        for layer in self.text_encoder:
            txt = layer(txt)

        for layer in self.pcd_encoder:
            point_cloud = layer(point_cloud)

        # Recover and constrain the curvature c
        c = torch.exp(self.log_c).clamp(0.1, 10.0)
        # Recover the scaling factor α
        alpha_txt = torch.exp(self.log_alpha_txt)
        alpha_pcd = torch.exp(self.log_alpha_pcd)
        # Scale the Euclidean features before projecting to hyperbolic space
        txt_scaled = txt * alpha_txt
        pcd_scaled = point_cloud * alpha_pcd

        txt_global_h = self.text_aggregator(txt_scaled, c=c)
        pcd_global_h = self.pcd_aggregator(pcd_scaled, c=c)

        if self.training:
            # During training, calculate and return the total loss
            if pc_indices is None:
                raise ValueError("pc_indices must be provided during training.")
            return self.loss_fn(txt_global_h, pcd_global_h, pc_indices, c)
        else:
            # During evaluation, calculate and return the similarity matrix

            # similarity_matrix = -self.lorentz_distance(txt_global_h.unsqueeze(1), pcd_global_h.unsqueeze(0), c)
            similarity_matrix = self.loss_fn.lorentz_inner_product(
                txt_global_h.unsqueeze(1), pcd_global_h.unsqueeze(0)
            )
            return similarity_matrix
