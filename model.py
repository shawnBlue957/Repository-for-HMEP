import torch
import torch.nn as nn
import torch.nn.functional as F
import clip


class FusionStrategies:
    CAT = 'cat'
    SUM = 'sum'
    GATE = 'gate'
    ATTENTION = 'attention'
    MH_ATTENTION = 'multihead_attention'


class FeatureFusion(nn.Module):
    """
    多模态特征融合模块，支持cat/sum/gate/attention/multihead_attention，含残差和深融合
    """

    def __init__(
            self,
            in_dims,
            out_dim,
            fusion_type='cat',
            hidden_dim=512,
            dropout_p=0.3,
            n_heads=4,  # 多头数量
            deep_layers=2  # 深度融合层数
    ):
        super().__init__()
        self.fusion_type = fusion_type
        self.in_dims = in_dims
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.deep_layers = deep_layers

        self.projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout_p)
            )
            for in_dim in in_dims
        ])

        if fusion_type == FusionStrategies.CAT:
            self.fusion_mlp = nn.Sequential(
                nn.Linear(hidden_dim * len(in_dims), hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout_p),
                nn.Linear(hidden_dim, out_dim)
            )
        elif fusion_type == FusionStrategies.SUM:
            self.weights = nn.Parameter(torch.ones(len(in_dims)))
            self.fusion_mlp = nn.Sequential(
                nn.Linear(hidden_dim, out_dim)
            )
        elif fusion_type == FusionStrategies.GATE:
            self.gate_linear = nn.Linear(hidden_dim * len(in_dims), len(in_dims))
            self.fusion_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout_p),
                nn.Linear(hidden_dim, out_dim)
            )
        elif fusion_type == FusionStrategies.ATTENTION:
            # 单头注意力
            self.attn_query = nn.Linear(hidden_dim, hidden_dim)
            self.attn_key = nn.Linear(hidden_dim, hidden_dim)
            self.attn_value = nn.Linear(hidden_dim, hidden_dim)
            self.fusion_mlp = nn.Sequential(
                nn.Linear(hidden_dim, out_dim)
            )
        elif fusion_type == FusionStrategies.MH_ATTENTION:
            # 多头注意力
            self.attn = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=n_heads,
                batch_first=True,
                dropout=dropout_p
            )
            # 深层残差融合块
            deep_layers = []
            for _ in range(self.deep_layers):
                deep_layers.append(nn.Linear(hidden_dim, hidden_dim))
                deep_layers.append(nn.ReLU())
                deep_layers.append(nn.LayerNorm(hidden_dim))
                deep_layers.append(nn.Dropout(dropout_p))
            self.deep_mlp = nn.Sequential(*deep_layers)
            self.final_proj = nn.Linear(hidden_dim, out_dim)
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

        self.output_norm = nn.LayerNorm(out_dim)

    def forward(self, features):
        # features: List[Tensor], 每个shape为(N, in_dim)
        proj_feats = [proj(fea) for proj, fea in zip(self.projs, features)]  # 每个(N, hidden_dim)

        if self.fusion_type == FusionStrategies.CAT:
            fused = torch.cat(proj_feats, dim=-1)
            fused = self.fusion_mlp(fused)
        elif self.fusion_type == FusionStrategies.SUM:
            w = F.softmax(self.weights, dim=0)
            fused = sum(w[i] * proj_feats[i] for i in range(len(proj_feats)))
            fused = self.fusion_mlp(fused)
        elif self.fusion_type == FusionStrategies.GATE:
            gate_logits = self.gate_linear(torch.cat(proj_feats, dim=-1))
            gates = torch.sigmoid(gate_logits)
            fused = sum(gates[:, i:i + 1] * proj_feats[i] for i in range(len(proj_feats)))
            fused = self.fusion_mlp(fused)
        elif self.fusion_type == FusionStrategies.ATTENTION:
            # 单头attention
            Q = self.attn_query(proj_feats[0]).unsqueeze(1)  # (N,1,H)
            K = torch.stack([self.attn_key(f) for f in proj_feats], dim=1)  # (N,M,H)
            V = torch.stack([self.attn_value(f) for f in proj_feats], dim=1)  # (N,M,H)
            attn = F.softmax((Q * K).sum(-1), dim=1).unsqueeze(-1)  # (N,M,1)
            fused = (attn * V).sum(1)  # (N,H)
            fused = self.fusion_mlp(fused)
        elif self.fusion_type == FusionStrategies.MH_ATTENTION:
            # 多头注意力，先堆成(N, M, H)
            x = torch.stack(proj_feats, dim=1)  # (N, M, H)
            out, _ = self.attn(x, x, x, need_weights=False)
            fused = x + out
            fused = fused.mean(dim=1)  # (N, H)
            fused = self.deep_mlp(fused) + fused  # 残差
            fused = self.final_proj(fused)
        else:
            raise NotImplementedError
        return self.output_norm(fused), proj_feats


class VGRelationModel(nn.Module):
    def __init__(
            self,
            num_predicates,
            clip_model_type="ViT-B/32",
            visual_feature_dim=512,
            text_feature_dim=512,
            coord_feature_dim=4,
            fused_feature_dim=256,
            dropout_p=0.3,
            fusion_type='cat',
            use_visual=True,
            use_text=True,
            use_coord=True,
            fusion_heads=4,
            fusion_depth=2
    ):
        super().__init__()
        self.num_predicates = num_predicates

        self.clip_model, _ = clip.load(clip_model_type)
        self._freeze_clip()

        input_dims = []
        self.modal_list = []
        if use_visual:
            input_dims.append(visual_feature_dim)
            self.modal_list.append('visual')
        if use_text:
            input_dims.append(text_feature_dim)
            self.modal_list.append('text')
        if use_coord:
            input_dims.append(coord_feature_dim)
            self.modal_list.append('coord')

        self.sub_fusion = FeatureFusion(
            input_dims, fused_feature_dim, fusion_type,
            dropout_p=dropout_p, n_heads=fusion_heads, deep_layers=fusion_depth)
        self.obj_fusion = FeatureFusion(
            input_dims, fused_feature_dim, fusion_type,
            dropout_p=dropout_p, n_heads=fusion_heads, deep_layers=fusion_depth)
        self.union_fusion = FeatureFusion(
            input_dims, fused_feature_dim, fusion_type,
            dropout_p=dropout_p, n_heads=fusion_heads, deep_layers=fusion_depth)

        self.alpha_predictor_mlp = nn.Sequential(
            nn.Linear(fused_feature_dim * 3, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(dropout_p),
            nn.Linear(128, 1)
        )
        self.img_feat_projector_for_gate = nn.Sequential(
            nn.Linear(visual_feature_dim, fused_feature_dim),
            nn.Dropout(dropout_p)
        )
        self.gate_activation = nn.Sigmoid()

        # relation_mlp的输入维度要与fused_feature_dim一致
        relation_layers = [
            nn.Linear(fused_feature_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(dropout_p),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(dropout_p)
        ]
        self.relation_mlp = nn.Sequential(*relation_layers)
        self.classifier = nn.Sequential(
            nn.Linear(128, self.num_predicates),
            nn.Dropout(dropout_p)
        )

    def _freeze_clip(self):
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.clip_model.eval()

    def _compute_union_box(self, sub_boxes, obj_boxes):
        sub_x1 = sub_boxes[:, 0]
        sub_y1 = sub_boxes[:, 1]
        sub_x2 = sub_boxes[:, 0] + sub_boxes[:, 2]
        sub_y2 = sub_boxes[:, 1] + sub_boxes[:, 3]
        obj_x1 = obj_boxes[:, 0]
        obj_y1 = obj_boxes[:, 1]
        obj_x2 = obj_boxes[:, 0] + obj_boxes[:, 2]
        obj_y2 = obj_boxes[:, 1] + obj_boxes[:, 3]
        union_x1 = torch.minimum(sub_x1, obj_x1)
        union_y1 = torch.minimum(sub_y1, obj_y1)
        union_x2 = torch.maximum(sub_x2, obj_x2)
        union_y2 = torch.maximum(sub_y2, obj_y2)
        return torch.stack([
            union_x1, union_y1, union_x2 - union_x1, union_y2 - union_y1
        ], dim=1)

    def _encode_phrase(self, phrases, device):
        with torch.no_grad():
            text_tokens = clip.tokenize(phrases).to(device)
            return self.clip_model.encode_text(text_tokens).float()

    def forward(self, batch, return_features=False):
        all_logits = []
        all_features = []  # 新增
        device = next(self.parameters()).device
        batch_size = len(batch['image_id'])
        for i in range(batch_size):
            img_data = {k: v[i] for k, v in batch.items()}
            sub_inputs, obj_inputs, union_inputs = [], [], []

            for modal in self.modal_list:
                if modal == 'visual':
                    sub_inputs.append(img_data['sub_visual_feats'].to(device))
                    obj_inputs.append(img_data['obj_visual_feats'].to(device))
                    union_inputs.append(img_data['union_visual_feats'].to(device))
                elif modal == 'text':
                    sub_inputs.append(F.normalize(img_data['sub_text_feats'].float().to(device), dim=1))
                    obj_inputs.append(F.normalize(img_data['obj_text_feats'].float().to(device), dim=1))
                    phrases = img_data['phrases']
                    phrase_text_feats_input = F.normalize(self._encode_phrase(phrases, device=device), dim=1)
                    union_inputs.append(phrase_text_feats_input)
                elif modal == 'coord':
                    sub_inputs.append(img_data['sub_boxes'].float().to(device))
                    obj_inputs.append(img_data['obj_boxes'].float().to(device))
                    union_boxes = self._compute_union_box(
                        img_data['sub_boxes'].float().to(device),
                        img_data['obj_boxes'].float().to(device)
                    )
                    union_inputs.append(union_boxes)

            sub_feat, sub_proj_feats = self.sub_fusion(sub_inputs)
            obj_feat, obj_proj_feats = self.obj_fusion(obj_inputs)
            union_feat, union_proj_feats = self.union_fusion(union_inputs)

            # 残差连接（仅单模态时才加）
            if len(sub_proj_feats) == 1:
                sub_feat = sub_feat + sub_proj_feats[0]
            if len(obj_proj_feats) == 1:
                obj_feat = obj_feat + obj_proj_feats[0]
            if len(union_proj_feats) == 1:
                union_feat = union_feat + union_proj_feats[0]

            global_img_feat = F.normalize(img_data['image'].float().unsqueeze(0).to(device), dim=-1)
            alpha_predictor_input = torch.cat([union_feat, sub_feat, obj_feat], dim=-1)
            alpha_logits = self.alpha_predictor_mlp(alpha_predictor_input)
            alpha = torch.sigmoid(alpha_logits)
            beta = 1.0 - alpha
            relation_feat_intermediate = union_feat - alpha * sub_feat - beta * obj_feat

            img_projection_for_gate = self.img_feat_projector_for_gate(global_img_feat).expand_as(
                relation_feat_intermediate)
            gate = self.gate_activation(img_projection_for_gate)
            contextualized_relation_feat = relation_feat_intermediate * gate

            final_relation_output = self.relation_mlp(contextualized_relation_feat)
            logits = self.classifier(final_relation_output)  # (N, num_predicates)
            all_logits.append(logits)

            if return_features:
                # 只保留当前样本所有特征
                # f_sub, f_obj, f_union, alpha, beta, f_pred_raw
                # 其中 f_pred_raw = relation_feat_intermediate
                # shape: (N, D)
                all_features.append({
                    'f_sub': sub_feat.detach(),
                    'f_obj': obj_feat.detach(),
                    'f_union': union_feat.detach(),
                    'alpha': alpha.detach(),
                    'beta': beta.detach(),
                    'f_pred_raw': relation_feat_intermediate.detach()
                })
        if return_features:
            return all_logits, all_features
        else:
            return all_logits
    def predict_triplets(self, batch, device=None):
        if device is None:
            device = next(self.parameters()).device
        self.eval()
        results = []
        with torch.no_grad():
            logits_list = self.forward(batch)
            batch_size = len(batch['image_id'])
            for i in range(batch_size):
                logits = logits_list[i]
                scores = F.softmax(logits, dim=1)
                pred_scores, pred_classes = torch.max(scores, dim=1)
                triplets = []
                N = logits.shape[0]
                gt_triplets = batch['gt_triplets'][i]
                # print(f"第{i}张图像: GT三元组数量: {len(gt_triplets)}, logits数(N): {N}")
                for j in range(N):
                    gt_triplet = gt_triplets[j]
                    sub_cls = gt_triplet['sub_cls']
                    obj_cls = gt_triplet['obj_cls']
                    sub_box = gt_triplet['sub_box']
                    obj_box = gt_triplet['obj_box']
                    pred_cls = pred_classes[j].item()
                    score = pred_scores[j].item()
                    triplets.append({
                        'sub_cls': sub_cls,
                        'obj_cls': obj_cls,
                        'pred_cls': pred_cls,
                        'sub_box': sub_box,
                        'obj_box': obj_box,
                        'score': score
                    })
                results.append(triplets)
        return results
