from tqdm import tqdm


def to_numpy(data):
    return data.detach().cpu().numpy()


def exp_weight_average(curr_val, prev_val, alpha=0.9, from_torch=True):
    if from_torch:
        curr_val = to_numpy(curr_val)
    return alpha * prev_val + (1 - alpha) * curr_val


def get_trainval_pbar(dataloader, epoch, epochs):

    pbar = tqdm(
        total=len(dataloader),
        leave=True,
        ncols=0,
        desc=f"Epoch {epoch+1}/{epochs}")

    return pbar


def get_predict_pbar(dataloader):
    pbar = tqdm(
        total=len(dataloader),
        leave=True,
        ncols=0,
        desc="Predict")

    return pbar


def extended_postfix(postfix, dct):
    postfixes = [postfix] + [f"{k}={v:.4f}" for k, v in dct.items()]
    return ", ".join(postfixes)


def update_epoch_metrics(target, preds, metrics, epoch_metrics):
    for m in metrics:
        value = m(target, preds)
        epoch_metrics[m.__name__] += value

