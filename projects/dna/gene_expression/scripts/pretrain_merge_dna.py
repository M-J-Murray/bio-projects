

def run_training():
    for epoch in range(1, epochs + 1):
        # Run training
        model.train()

        epoch_metrics = []
        for batch_idx, (label_ids, padding_mask, seq_lengths) in enumerate(train_loader):
            if batch_idx >= batches_per_epoch:
                break
            label_ids, padding_mask, seq_lengths = (
                label_ids.to(device),
                padding_mask.to(device),
                seq_lengths.to(device),
            )

            # Full Reconstruction loss
            # TODO: sample local reduction from normal dist to ensure 40% - 60% local reduction.
            logits, _ = model(label_ids, padding_mask, local_reduction=local_reduction)
            loss1 = F.cross_entropy(logits.flatten(0, 1), label_ids.flatten(0, 1)) / seq_lengths.sum()
            full_acc = (logits.argmax(dim=-1) == label_ids).float().mean()

            # Latent Reconstruction loss
            logits, latent_s = model(
                label_ids, padding_mask, local_reduction=local_reduction, latent_reduction=latent_reduction
            )
            loss2 = F.cross_entropy(logits.flatten(0, 1), label_ids.flatten(0, 1)) / seq_lengths.sum()
            latent_acc = (logits.argmax(dim=-1) == label_ids).float().mean()

            # MTM loss
            token_ids, mask_matrix = apply_masking(
                label_ids, seq_lengths, latent_s, tokenizer.char_to_id[tokenizer.MASK]
            )
            logits, _ = model(token_ids, padding_mask, local_reduction=local_reduction)
            loss3 = F.cross_entropy(logits[mask_matrix], label_ids[mask_matrix]) / mask_matrix.sum()
            mask_acc = (logits.argmax(dim=-1)[mask_matrix] == label_ids[mask_matrix]).float().mean()
            loss = loss1 + lerl_weight * loss2 + loss3

            epoch_metrics.append(
                {
                    "loss1": loss1.item(),
                    "loss2": loss2.item(),
                    "loss3": loss3.item(),
                    "total_loss": loss.item(),
                    "full_acc": full_acc.item(),
                    "latent_acc": latent_acc.item(),
                    "mask_acc": mask_acc.item(),
                    "samples": len(label_ids),
                }
            )
            print("Step " + ", ".join(f"{k}: {v:.5f}" for k, v in epoch_metrics[-1].items()))

            optim.zero_grad()
            loss.backward()
            optim.step()

        epoch_loss = sum(item["total_loss"] for item in epoch_metrics)
        epoch_acc = sum(item["mask_acc"] * item["samples"] for item in epoch_metrics)
        epoch_samples = sum(item["samples"] for item in epoch_metrics)
        print(f"Epoch loss: {epoch_loss:.5f}, mask_acc: {epoch_acc / epoch_samples:.2%} ")

        # Check validation
        if epoch % epochs_per_val != 0:
            continue

        #  Run validation
        val_metrics = []
        model.eval()
        for label_ids, padding_mask, seq_lengths in val_loader:
            label_ids, padding_mask, seq_lengths = (
                label_ids.to(device),
                padding_mask.to(device),
                seq_lengths.to(device),
            )
            with torch.no_grad():
                logits, _ = model(label_ids, padding_mask, local_reduction=local_reduction)
            loss = F.cross_entropy(logits.flatten(0, 1), label_ids.flatten(0, 1)) / seq_lengths.sum()
            full_acc = (logits.argmax(dim=-1) == label_ids).float().mean()
            val_metrics.append(
                {
                    "loss": loss.item(),
                    "acc": full_acc.item(),
                    "samples": len(label_ids),
                }
            )

        val_loss = sum(item["loss"] for item in val_metrics)
        val_acc = sum(item["acc"] * item["samples"] for item in val_metrics)
        val_samples = sum(item["samples"] for item in val_metrics)
        print(f"Val loss: {val_loss:.5f}, full_acc: {val_acc / val_samples:.2%}")


run_training()

def main():
    # Load datasets
    hf_train = load_dataset("InstaDeepAI/multi_species_genomes", trust_remote_code=True, split="train", streaming=True)
    hf_val = load_dataset("InstaDeepAI/multi_species_genomes", trust_remote_code=True, split="validation", streaming=True)
    train_samples = list(map(lambda x: x["sequence"], hf_train.take(10_000)))
    val_samples = list(map(lambda x: x["sequence"], hf_train.take(1_000)))
    print(f"Downloaded {len(train_samples)} training samples, and {len(val_samples)} validation samples.")

    # Training and validation
    seed = 123
    epochs = 2
    batches_per_epoch = 10
    epochs_per_val = 1
    train_batch_size = 64
    val_batch_size = 256
    sequence_len = 256
    local_reduction = 64
    latent_reduction = 16
    lr = 1e-4
    lerl_weight = 0.25
    accelerator = "mps"  # cpu, gpu, mps

    torch.manual_seed(seed)

    # Validated that when full model is created it has 380mil parameters
    tokenizer = MtmDnaTokenizer()
    model = MergeDNAModel(
        local_encoder_blocks=2,
        latent_encoder_blocks=10,
        latent_decoder_blocks=4,
        local_decoder_blocks=2,
        vocab_size=tokenizer.vocab_size,
        embedding_dims=256,
        num_heads=4,
        window_size=16,
        top_k=3,
        temperature=0.1,
    )

    # Total parameters (including frozen ones)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,}")

    # model = torch.compile(model) # TODO: work out why this stopped working
    device = torch.device("cuda") if accelerator == "gpu" else torch.device(accelerator)
    model = model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=1e-8)

    train_dataset = NtMsGenomeDataset(train_samples, sequence_len, tokenizer)
    val_dataset = NtMsGenomeDataset(val_samples, sequence_len, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)

    run_training()

if __name__ == "__main__":
    main()