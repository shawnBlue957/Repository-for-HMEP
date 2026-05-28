
import argparse
import os
import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from VRD_dataloder import VRDRelationDataset, collate_fn
from new_model import VGRelationModel


def invertibility_loss(features, pair_sample_num=None, reg_weight=0.01, margin=1.0):
  
    total_pair_loss = None
    total_pair_count = 0
    total_reg_loss = None
    total_reg_count = 0

    for feat in features:
        f_union = feat['f_union']          # (N, D)
        f_pred_raw = feat['f_pred_raw']    # (N, D)

        N = f_pred_raw.shape[0]
        if N == 0:
            continue

        pred_norm_sq = f_pred_raw.pow(2).sum(dim=1)
        reg_term = torch.relu(margin - pred_norm_sq).mean()
        total_reg_loss = reg_term if total_reg_loss is None else total_reg_loss + reg_term
        total_reg_count += 1

        if N < 2:
            continue

        pairs = []
        if pair_sample_num is not None and pair_sample_num < N * (N - 1) // 2:
            sampled = set(torch.randperm(N * (N - 1) // 2)[:pair_sample_num].tolist())
            cnt = 0
            for i in range(N):
                for j in range(i + 1, N):
                    if cnt in sampled:
                        pairs.append((i, j))
                    cnt += 1
        else:
            for i in range(N):
                for j in range(i + 1, N):
                    pairs.append((i, j))

        for i, j in pairs:
            left = f_pred_raw[i] + f_pred_raw[j]
            right = f_union[i] + f_union[j]
            loss_ij = torch.nn.functional.mse_loss(left, right)
            if total_pair_loss is None:
                total_pair_loss = loss_ij
            else:
                total_pair_loss = total_pair_loss + loss_ij
        total_pair_count += len(pairs)

    if total_pair_count == 0:
        device = features[0]['f_pred_raw'].device if features else torch.device('cpu')
        pair_loss = torch.tensor(0.0, device=device, requires_grad=True)
    else:
        pair_loss = total_pair_loss / total_pair_count

    if total_reg_count == 0:
        device = features[0]['f_pred_raw'].device if features else torch.device('cpu')
        reg_loss = torch.tensor(0.0, device=device, requires_grad=True)
    else:
        reg_loss = total_reg_loss / total_reg_count

    return pair_loss + reg_weight * reg_loss


def evaluate(model, dataloader, device, writer=None, epoch=None, phase="test",
             use_invertibility_loss=False, inv_reg_weight=0.01):
    model.eval()
    correct, loss_sum, num_samples = 0, 0.0, 0
    criterion = CrossEntropyLoss()
    inv_loss_sum = 0.0
    n_batches_inv = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating ({phase})", leave=False):
            if not batch:
                continue

            if use_invertibility_loss:
                batch_logits, all_features = model(batch, return_features=True)
                inv_loss = invertibility_loss(all_features, reg_weight=inv_reg_weight)
                if torch.isfinite(inv_loss):
                    inv_loss_sum += inv_loss.item()
                n_batches_inv += 1
            else:
                batch_logits = model(batch)

            batch_labels = batch.get('labels', None)
            if batch_labels is None:
                continue

            for logits, labels in zip(batch_logits, batch_labels):
                
                if labels is None or labels.numel() == 0 or logits.numel() == 0:
                    continue
                if not torch.isfinite(logits).all():
                    continue

                labels = labels.to(device)
                logits = logits.to(device)

                loss = criterion(logits, labels)
                if not torch.isfinite(loss):
                    continue

                _, preds = torch.max(logits, 1)
                correct += (preds == labels).sum().item()
                loss_sum += loss.item() * labels.size(0)
                num_samples += labels.size(0)

    avg_loss = loss_sum / num_samples if num_samples > 0 else 0.0
    acc = correct / num_samples if num_samples > 0 else 0.0
    avg_inv_loss = inv_loss_sum / n_batches_inv if (use_invertibility_loss and n_batches_inv > 0) else 0.0
    if writer is not None and epoch is not None:
        writer.add_scalar(f"{phase}/Loss", avg_loss, epoch)
        writer.add_scalar(f"{phase}/Acc", acc, epoch)
        if use_invertibility_loss:
            writer.add_scalar(f"{phase}/InvLoss", avg_inv_loss, epoch)
    return avg_loss, acc, avg_inv_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test'], default='train')
    parser.add_argument('--train_json', type=str, default=r'D:\cxaPythonWorkSpace\cxa\visual_relationship_detect-master\data\vrd\coco_vrd_json_dataset\annotations_train.json')
    parser.add_argument('--test_json', type=str, default=r'D:\cxaPythonWorkSpace\cxa\visual_relationship_detect-master\data\vrd\coco_vrd_json_dataset\annotations_test.json')
    parser.add_argument('--obj2id', type=str, default=r'D:\cxaPythonWorkSpace\cxa\visual_relationship_detect-master\data\vrd\coco_vrd_json_dataset\objects2id.json')
    parser.add_argument('--pred2id', type=str, default=r'D:\cxaPythonWorkSpace\cxa\visual_relationship_detect-master\data\vrd\coco_vrd_json_dataset\predicate2id.json')
    parser.add_argument('--img_dir', type=str, default=r'D:\cxaPythonWorkSpace\cxa\dataset\VRD\sg_train_images')
    parser.add_argument('--cache_dir', type=str, default=None)

    parser.add_argument('--clip_model_type', type=str, default='ViT-B/32')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--log_dir', type=str, default='runs/vrd_relation2')
    parser.add_argument('--save_dir', type=str, default='checkpoints_vrd333/vrd_inv_1.0')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--use_invertibility_loss', type=int, default=1)
    parser.add_argument('--lambda_inv', type=float, default=1.0)
    parser.add_argument('--inv_reg_weight', type=float, default=0.01)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_inv = bool(args.use_invertibility_loss)

    pred2id = load_json_from_arg(args.pred2id)
    num_predicates = len(pred2id)

    train_set = VRDRelationDataset(
        ann_file=args.train_json,
        img_dir_root=args.img_dir,
        clip_model_type=args.clip_model_type,
        device=device.type if isinstance(device, torch.device) else device,
        cache_dir=args.cache_dir,
        objects2id_path=args.obj2id,
        predicate2id_path=args.pred2id,
        fix_invalid_bbox=True
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn)

    model = VGRelationModel(
        num_predicates=num_predicates,
        clip_model_type=args.clip_model_type,
        visual_feature_dim=512,
        text_feature_dim=512,
        coord_feature_dim=4,
        fused_feature_dim=256,
        dropout_p=0.3,
        fusion_type='multihead_attention',
        fusion_heads=4,
        fusion_depth=2,
        use_visual=True,
        use_text=True,
        use_coord=True
    ).to(device)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    criterion = CrossEntropyLoss()
    writer = SummaryWriter(log_dir=args.log_dir)
    global_step = 0

    start_epoch = 0
    if args.resume and os.path.exists(args.save_dir):
        ckpts = [f for f in os.listdir(args.save_dir) if f.endswith('.pth')]
        if ckpts:
            ckpts.sort()
            latest = ckpts[-1]
            model.load_state_dict(torch.load(os.path.join(args.save_dir, latest), map_location=device))
            print(f"[Resume] loaded {latest}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        running_inv = 0.0
        correct = 0
        total_samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.epochs}]")
        for batch in pbar:
            if not batch:
                continue

            optimizer.zero_grad()
            if use_inv:
                batch_logits, all_features = model(batch, return_features=True)
                inv_loss = invertibility_loss(all_features, reg_weight=args.inv_reg_weight)
                if not isinstance(inv_loss, torch.Tensor):
                    inv_loss = torch.tensor(float(inv_loss), device=device)
                inv_loss = inv_loss.to(device)
                if not torch.isfinite(inv_loss):
                    inv_loss = torch.tensor(0.0, device=device)
            else:
                batch_logits = model(batch)
                inv_loss = torch.tensor(0.0, device=device)

            batch_labels = batch.get('labels', None)
            if batch_labels is None:
                continue

            total_loss = torch.tensor(0.0, device=device)
            batch_correct = 0
            batch_total = 0

            for logits, labels in zip(batch_logits, batch_labels):
               
                if labels is None or labels.numel() == 0 or logits.numel() == 0:
                    continue
                if not torch.isfinite(logits).all():
                    continue

                labels = labels.to(device)
                logits = logits.to(device)

                loss = criterion(logits, labels)
                if not torch.isfinite(loss):
                    continue

                total_loss = total_loss + loss * labels.size(0)
                _, preds = torch.max(logits, 1)
                batch_correct += (preds == labels).sum().item()
                batch_total += labels.size(0)

            if batch_total == 0:
                continue

            avg_loss = total_loss / batch_total
            train_obj = avg_loss + args.lambda_inv * inv_loss
            if torch.isfinite(train_obj):
                train_obj.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            running_loss += avg_loss.item() * batch_total
            running_inv += (inv_loss.item() if torch.isfinite(inv_loss) else 0.0) * batch_total
            correct += batch_correct
            total_samples += batch_total

            writer.add_scalar("Loss/train_step", avg_loss.item(), global_step)
            if use_inv:
                writer.add_scalar("InvLoss/train_step", inv_loss.item() if torch.isfinite(inv_loss) else 0.0, global_step)

            if (global_step + 1) % 20 == 0:
                acc = correct / total_samples if total_samples > 0 else 0.0
                pbar.set_postfix(loss=avg_loss.item(),
                                 inv_loss=(inv_loss.item() if torch.isfinite(inv_loss) else 0.0) if use_inv else 0.0,
                                 acc=acc)

            global_step += 1

        epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
        epoch_inv = running_inv / total_samples if total_samples > 0 else 0.0
        epoch_acc = correct / total_samples if total_samples > 0 else 0.0

        print(f"[Epoch {epoch+1}] loss: {epoch_loss:.4f} inv: {epoch_inv:.4f} acc: {epoch_acc:.4f}")
        writer.add_scalar("Loss/train_epoch", epoch_loss, epoch)
        writer.add_scalar("Acc/train_epoch", epoch_acc, epoch)
        if use_inv:
            writer.add_scalar("InvLoss/train_epoch", epoch_inv, epoch)

        os.makedirs(args.save_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(args.save_dir, f"vrd_relation_epoch{epoch+1}.pth"))

    writer.close()


def load_json_from_arg(path_or_str):
    if path_or_str is None:
        return {}
    if isinstance(path_or_str, dict):
        return path_or_str
    if isinstance(path_or_str, str) and os.path.exists(path_or_str):
        import json
        with open(path_or_str, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


if __name__ == '__main__':
    main()
