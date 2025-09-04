import argparse
import os
from dataloader import VGRelationDataset, collate_fn
from model import VGRelationModel
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def invertibility_loss(features, pair_sample_num=None):
    """
    features: List[dict], each dict contains 'f_sub', 'f_obj', 'f_union', 'alpha', 'beta', 'f_pred_raw'
    pair_sample_num: If None, take all pairs, otherwise randomly sample pair_sample_num pairs
    """
    total_loss = 0.0
    total_count = 0
    for feat in features:
        f_sub = feat['f_sub']  # (N, D)
        f_obj = feat['f_obj']  # (N, D)
        f_union = feat['f_union']  # (N, D)
        alpha = feat['alpha']  # (N, 1)
        beta = feat['beta']  # (N, 1)
        f_pred_raw = feat['f_pred_raw']  # (N, D)

        N = f_sub.shape[0]
        if N < 2:
            continue  # Skip samples that cannot form a pair
        #Default is all pairs, or sample pair_sample_num pairs
        pairs = []
        if pair_sample_num is not None and pair_sample_num < N * (N - 1) // 2:
            idx = torch.randperm(N * (N - 1) // 2)[:pair_sample_num]
            cnt = 0
            for i in range(N):
                for j in range(i + 1, N):
                    if cnt in idx:
                        pairs.append((i, j))
                    cnt += 1
        else:
            for i in range(N):
                for j in range(i + 1, N):
                    pairs.append((i, j))
        for i, j in pairs:
            left = f_pred_raw[i] + f_pred_raw[j]
            right = f_union[i] + f_union[j] - (
                        alpha[i] * f_sub[i] + alpha[j] * f_sub[j] + beta[i] * f_obj[i] + beta[j] * f_obj[j])
            total_loss += F.mse_loss(left, right)
        total_count += len(pairs)
    if total_count == 0:
        return torch.tensor(0.0, requires_grad=True)
    return total_loss / total_count


def evaluate(model, dataloader, device, writer=None, epoch=None, phase="val", use_invertibility_loss=False,
             lambda_inv=1.0):
    model.eval()
    correct, loss_sum, num_samples = 0, 0.0, 0
    criterion = CrossEntropyLoss()
    all_preds, all_labels = [], []
    inv_loss_sum = 0.0  
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating ({phase})", leave=False):
            if use_invertibility_loss:
                batch_logits, all_features = model(batch, return_features=True)
                inv_loss = invertibility_loss(all_features)
                inv_loss_sum += inv_loss.item()
            else:
                batch_logits = model(batch)
            batch_labels = batch['labels']
            for logits, labels in zip(batch_logits, batch_labels):
                labels = labels.to(device)
                logits = logits.to(device)
                loss = criterion(logits, labels)
                _, preds = torch.max(logits, 1)
                correct += (preds == labels).sum().item()
                batch_size = labels.size(0)
                loss_sum += loss.item() * batch_size
                num_samples += batch_size
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
    acc = correct / num_samples if num_samples > 0 else 0
    avg_loss = loss_sum / num_samples if num_samples > 0 else 0.0
    if use_invertibility_loss and len(dataloader) > 0:
        avg_inv_loss = inv_loss_sum / len(dataloader)
    else:
        avg_inv_loss = 0.0
    return avg_loss, acc, avg_inv_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'val', 'test'], default='train')
    parser.add_argument('--train_json', type=str, default='dataset/vg/annotations/instances_vg_train.json')
    parser.add_argument('--val_json', type=str, default='dataset/vg/annotations/instances_vg_test_val.json')
    parser.add_argument('--test_json', type=str, default='dataset/vg/annotations/instances_vg_test_new_test.json')
    parser.add_argument('--object_json', type=str, default='dataset/vg/annotations/objects.json')
    parser.add_argument('--img_dir', type=str, default='dataset/vg/images')
    parser.add_argument('--cache_dir', type=str, default='feature_cache')

    parser.add_argument('--clip_model_type', type=str, default='ViT-B/32')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--min_lr', type=float, default=1e-6, help="Minimum learning rate for scheduler")
    parser.add_argument('--lr_scheduler', type=str, choices=['step', 'cosine', 'none'], default='cosine',
                        help="Type of learning rate scheduler")
    parser.add_argument('--step_size', type=int, default=10, help="StepLR step size")
    parser.add_argument('--gamma', type=float, default=0.5, help="StepLR gamma")
    parser.add_argument('--num_workers', type=int, default=0)

    parser.add_argument('--log_dir', type=str, default='runs/vg_relation/multihead_attention_v2')
    parser.add_argument('--save_dir', type=str, default='checkpoints_multihead_attention_v2')
    parser.add_argument('--resume', action='store_true', help="Resume training from last checkpoint")
    parser.add_argument('--early_stop_patience', type=int, default=5, help="Patience for early stopping")
    parser.add_argument('--dropout_p', type=float, default=0.3, help="Dropout probability")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="L2 regularization")
    # 模型结构相关超参数
    parser.add_argument('--visual_feature_dim', type=int, default=512)
    parser.add_argument('--text_feature_dim', type=int, default=512)
    parser.add_argument('--coord_feature_dim', type=int, default=4)
    parser.add_argument('--fused_feature_dim', type=int, default=256)
    parser.add_argument('--fusion_type', type=str, default='multihead_attention',
                        choices=['cat', 'sum', 'gate', 'attention', 'multihead_attention'],
                        help='Feature fusion method')
    parser.add_argument('--fusion_heads', type=int, default=4, help='Number of multi-head attention heads')
    parser.add_argument('--fusion_depth', type=int, default=2, help='Fusion Depth (Multi-layer MLP)')
    parser.add_argument('--use_visual', type=int, default=1, help='Whether to use visual features')
    parser.add_argument('--use_text', type=int, default=1, help='Whether to use text features')
    parser.add_argument('--use_coord', type=int, default=1, help='Whether to use coordinate features')
    # 新增：可逆性损失相关参数
    parser.add_argument('--use_invertibility_loss', type=int, default=1, help="Whether to enable reversible loss function")
    parser.add_argument('--lambda_inv', type=float, default=1.0, help="Reversibility loss function weight")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_visual = bool(args.use_visual)
    use_text = bool(args.use_text)
    use_coord = bool(args.use_coord)
    use_invertibility_loss = bool(args.use_invertibility_loss)
    lambda_inv = args.lambda_inv

    model_kwargs = dict(
        num_predicates=None,  
        clip_model_type=args.clip_model_type,
        visual_feature_dim=args.visual_feature_dim,
        text_feature_dim=args.text_feature_dim,
        coord_feature_dim=args.coord_feature_dim,
        fused_feature_dim=args.fused_feature_dim,
        dropout_p=args.dropout_p,
        fusion_type=args.fusion_type,
        fusion_heads=args.fusion_heads,
        fusion_depth=args.fusion_depth,
        use_visual=use_visual,
        use_text=use_text,
        use_coord=use_coord
    )

    if args.mode == 'train':
        train_set = VGRelationDataset(
            ann_file=args.train_json, img_dir_root=args.img_dir,
            object_json=args.object_json, clip_model_type=args.clip_model_type,
            device=device, cache_dir=args.cache_dir
        )
        val_set = VGRelationDataset(
            ann_file=args.val_json, img_dir_root=args.img_dir,
            object_json=args.object_json, clip_model_type=args.clip_model_type,
            device=device, cache_dir=args.cache_dir
        )
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                  collate_fn=collate_fn)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                collate_fn=collate_fn)

        model_kwargs['num_predicates'] = len(train_set.relationships)
        model = VGRelationModel(**model_kwargs).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = CrossEntropyLoss()
        writer = SummaryWriter(log_dir=args.log_dir)
        global_step = 0

        if args.lr_scheduler == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        elif args.lr_scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
        else:
            scheduler = None

        start_epoch = 0
        best_val_loss = float('inf')
        epochs_no_improve = 0

        if args.resume:
            last_ckpts = [f for f in os.listdir(args.save_dir) if f.startswith("vg_relation_epoch")]
            if last_ckpts:
                last_ckpts.sort(key=lambda x: int(x.split("epoch")[1].split(".")[0]))
                latest_ckpt = last_ckpts[-1]
                checkpoint_path = os.path.join(args.save_dir, latest_ckpt)
                print(f"Resuming from checkpoint: {checkpoint_path}")
                state_dict = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(state_dict)
                start_epoch = int(latest_ckpt.split("epoch")[1].split(".")[0])
            else:
                print("No checkpoint found, training from scratch.")

        for epoch in range(start_epoch, args.epochs):
            model.train()
            running_loss, running_inv_loss, correct, num_samples = 0.0, 0.0, 0, 0
            all_preds, all_labels = [], []
            pbar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{args.epochs}]")
            for i, batch in enumerate(pbar):
                optimizer.zero_grad()
                if use_invertibility_loss:
                    batch_logits, all_features = model(batch, return_features=True)
                    inv_loss = invertibility_loss(all_features)
                else:
                    batch_logits = model(batch)
                    inv_loss = torch.tensor(0.0, device=device)
                batch_labels = batch['labels']
                total_loss = 0.0
                batch_correct = 0
                batch_total = 0
                for logits, labels in zip(batch_logits, batch_labels):
                    labels = labels.to(device)
                    logits = logits.to(device)
                    loss = criterion(logits, labels)
                    total_loss += loss * labels.size(0)
                    _, preds = torch.max(logits, 1)
                    batch_correct += (preds == labels).sum().item()
                    batch_total += labels.size(0)
                    all_preds.append(preds.cpu())
                    all_labels.append(labels.cpu())
                if batch_total > 0:
                    avg_loss = total_loss / batch_total
                else:
                    avg_loss = total_loss
                total_train_loss = avg_loss + lambda_inv * inv_loss
                total_train_loss.backward()
                optimizer.step()

                running_loss += avg_loss.item() * batch_total
                running_inv_loss += inv_loss.item() * batch_total
                correct += batch_correct
                num_samples += batch_total
                writer.add_scalar("Loss/train_step", avg_loss.item(), global_step)
                if use_invertibility_loss:
                    writer.add_scalar("InvLoss/train_step", inv_loss.item(), global_step)
                if (i + 1) % 20 == 0:
                    acc = correct / num_samples if num_samples > 0 else 0
                    writer.add_scalar("Acc/train_step", acc, global_step)
                    pbar.set_postfix(loss=avg_loss.item(), inv_loss=inv_loss.item() if use_invertibility_loss else 0.0,
                                     acc=acc)
                global_step += 1
            epoch_loss = running_loss / num_samples if num_samples > 0 else 0.0
            epoch_inv_loss = running_inv_loss / num_samples if num_samples > 0 else 0.0
            epoch_acc = correct / num_samples if num_samples > 0 else 0
            writer.add_scalar("Loss/train_epoch", epoch_loss, epoch)
            if use_invertibility_loss:
                writer.add_scalar("InvLoss/train_epoch", epoch_inv_loss, epoch)
            writer.add_scalar("Acc/train_epoch", epoch_acc, epoch)
            print(
                f"[Epoch {epoch + 1}] Train Loss: {epoch_loss:.4f} InvLoss: {epoch_inv_loss:.4f} Acc: {epoch_acc:.4f}")

            val_loss, val_acc, val_inv_loss = evaluate(
                model, val_loader, device, writer=writer, epoch=epoch, phase="val",
                use_invertibility_loss=use_invertibility_loss, lambda_inv=lambda_inv
            )
            writer.add_scalar("Loss/val_epoch", val_loss, epoch)
            if use_invertibility_loss:
                writer.add_scalar("InvLoss/val_epoch", val_inv_loss, epoch)
            writer.add_scalar("Acc/val_epoch", val_acc, epoch)
            print(f"[Epoch {epoch + 1}] Val   Loss: {val_loss:.4f} InvLoss: {val_inv_loss:.4f} Acc: {val_acc:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                if not os.path.exists(args.save_dir):
                    os.makedirs(args.save_dir)
                torch.save(model.state_dict(), os.path.join(args.save_dir, "vg_relation_best.pth"))
            else:
                epochs_no_improve += 1

            if scheduler is not None:
                scheduler.step()
                writer.add_scalar("LR", optimizer.param_groups[0]['lr'], epoch)

            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            torch.save(model.state_dict(), f"{args.save_dir}/vg_relation_epoch{epoch + 1}.pth")

            if epochs_no_improve >= args.early_stop_patience:
                print(f"Early stopping at epoch {epoch + 1} (no improvement for {args.early_stop_patience} epochs)")
                break

        writer.close()

    elif args.mode == 'val':
        val_set = VGRelationDataset(
            ann_file=args.val_json, img_dir_root=args.img_dir,
            object_json=args.object_json, clip_model_type=args.clip_model_type,
            device=device, cache_dir=args.cache_dir
        )
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                collate_fn=collate_fn)
        model_kwargs['num_predicates'] = len(val_set.relationships)
        model = VGRelationModel(**model_kwargs).to(device)
        model.load_state_dict(torch.load(os.path.join(args.save_dir, "vg_relation_best.pth"), map_location=device))
        writer = SummaryWriter(log_dir=args.log_dir)
        val_loss, val_acc, val_inv_loss = evaluate(
            model, val_loader, device, writer=writer, epoch=0, phase="val",
            use_invertibility_loss=use_invertibility_loss, lambda_inv=lambda_inv
        )
        print(f"Val Loss: {val_loss:.4f} InvLoss: {val_inv_loss:.4f} Acc: {val_acc:.4f}")
        writer.close()

    elif args.mode == 'test':
        test_set = VGRelationDataset(
            ann_file=args.test_json, img_dir_root=args.img_dir,
            object_json=args.object_json, clip_model_type=args.clip_model_type,
            device=device, cache_dir=args.cache_dir
        )
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                 collate_fn=collate_fn)
        model_kwargs['num_predicates'] = len(test_set.relationships)
        model = VGRelationModel(**model_kwargs).to(device)
        model.load_state_dict(torch.load(os.path.join(args.save_dir, "vg_relation_best.pth"), map_location=device))
        writer = SummaryWriter(log_dir=args.log_dir)
        test_loss, test_acc, test_inv_loss = evaluate(
            model, test_loader, device, writer=writer, epoch=0, phase="test",
            use_invertibility_loss=use_invertibility_loss, lambda_inv=lambda_inv
        )
        print(f"Test Loss: {test_loss:.4f} InvLoss: {test_inv_loss:.4f} Acc: {test_acc:.4f}")
        writer.close()


if __name__ == "__main__":
    main()

