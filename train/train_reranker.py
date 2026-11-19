import argparse, hashlib, json, os, random, sys, time
import torch
import torch.nn.functional as F
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from retrieval.rerank import RERANK_MODEL, MAX_LEN, QUERY_MAX_TOKENS
from train.rr_groups import load_rows, epoch_groups

PASSAGES = "data/processed/passages.jsonl"
LR, WD, WARMUP_FRAC, EPOCHS, SEED = 2e-5, 0.01, 0.05, 3, 28
GROUPS_PER_STEP, MICRO_GROUPS, CLIP = 8, 2, 1.0


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None)
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--wandb", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        out = os.path.join("models", "smoke", "rr")
    else:
        if not a.tag:
            sys.exit("--tag required")
        out = os.path.join("models", "reranker", a.tag)
    if os.path.exists(out) and os.listdir(out):
        sys.exit(out + " is not empty, move it aside by hand")

    with open("data/processed/index/dense_ft.manifest.json", encoding="utf-8") as f:
        if json.load(f)["passages_sha256"] != sha256_file(PASSAGES):
            sys.exit("passages.jsonl does not match the manifest")
    text = {}
    with open(PASSAGES, encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            text[p["passage_id"]] = p["text"]
    rows = load_rows()
    qtext = {r["qid"]: r["query"] for r in rows}

    tok = AutoTokenizer.from_pretrained(RERANK_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(RERANK_MODEL).to("cuda")
    model.train()

    def cap_query(q):
        ids = tok(q, add_special_tokens=False)["input_ids"]
        return tok.decode(ids[:QUERY_MAX_TOKENS]) if len(ids) > QUERY_MAX_TOKENS else q

    steps_per_epoch = len(epoch_groups(rows, 0)) // GROUPS_PER_STEP
    total = a.smoke if a.smoke else steps_per_epoch * EPOCHS
    nd = ["bias", "LayerNorm.weight"]
    params = [{"params": [p for n, p in model.named_parameters() if not any(x in n for x in nd)],
               "weight_decay": WD},
              {"params": [p for n, p in model.named_parameters() if any(x in n for x in nd)],
               "weight_decay": 0.0}]
    opt = torch.optim.AdamW(params, lr=LR)
    sched = get_linear_schedule_with_warmup(opt, int(WARMUP_FRAC * total), total)
    scaler = torch.amp.GradScaler("cuda")
    print("steps per epoch", steps_per_epoch, "| total steps", total,
          "| warmup", int(WARMUP_FRAC * total))

    if a.wandb:
        import wandb
        wandb.init(project="plainsight", name=a.tag or "rr-smoke",
                   config={"lr": LR, "wd": WD, "warmup": WARMUP_FRAC, "epochs": EPOCHS,
                           "groups_per_step": GROUPS_PER_STEP, "micro_groups": MICRO_GROUPS,
                           "clip": CLIP, "seed": SEED, "steps": total})

    step, pairs, losses = 0, 0, []
    t0 = t_steady = time.time()
    pairs_steady = 0
    log_every = 10 if a.smoke else 100
    done = False
    for epoch in range(EPOCHS):
        groups = epoch_groups(rows, epoch)
        random.Random("%d-shuffle-%d" % (SEED, epoch)).shuffle(groups)
        for s in range(steps_per_epoch):
            batch = groups[s * GROUPS_PER_STEP:(s + 1) * GROUPS_PER_STEP]
            step_loss = 0.0
            for m in range(0, GROUPS_PER_STEP, MICRO_GROUPS):
                qs, ps, sizes = [], [], []
                for qid, pos, negs in batch[m:m + MICRO_GROUPS]:
                    items = [pos] + negs
                    qs += [cap_query(qtext[qid])] * len(items)
                    ps += [text[i] for i in items]
                    sizes.append(len(items))
                enc = tok(qs, ps, truncation="only_second", max_length=MAX_LEN,
                          padding=True, return_tensors="pt").to("cuda")
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(**enc).logits[:, 0]
                logits = logits.float()
                zero = torch.zeros(1, dtype=torch.long, device="cuda")
                loss = sum(F.cross_entropy(l[None], zero) for l in logits.split(sizes)) / GROUPS_PER_STEP
                scaler.scale(loss).backward()
                step_loss += loss.item()
                pairs += len(ps)
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1
            losses.append(step_loss)
            if step == 5:
                t_steady, pairs_steady = time.time(), pairs
            if step % log_every == 0:
                ml = sum(losses[-log_every:]) / log_every
                print("step %d/%d epoch %d loss %.4f lr %.2e %.0f s" %
                      (step, total, epoch, ml, sched.get_last_lr()[0], time.time() - t0), flush=True)
                if a.wandb:
                    wandb.log({"loss": ml, "lr": sched.get_last_lr()[0]}, step=step)
            if a.smoke and step >= a.smoke:
                done = True
                break
        if done:
            break
        d = os.path.join(out, "epoch-%d" % (epoch + 1))
        os.makedirs(d)
        model.save_pretrained(d)
        tok.save_pretrained(d)
        print("saved", d, flush=True)

    secs = time.time() - t0
    print("steps", step, "pairs", pairs, "seconds %.0f" % secs)
    if step > 5:
        print("steady pairs/s %.1f" % ((pairs - pairs_steady) / (time.time() - t_steady)))
    print("peak allocated GB %.2f" % (torch.cuda.max_memory_allocated() / 1e9))
    if a.smoke:
        os.makedirs(out, exist_ok=True)
        model.save_pretrained(out)
        tok.save_pretrained(out)
        print("smoke checkpoint saved to", out)
        print("first 10 steps mean loss %.4f, last 10 %.4f" %
              (sum(losses[:10]) / 10, sum(losses[-10:]) / 10))


if __name__ == "__main__":
    main()