from random import choice
import torch
import argparse


def get_params():
    args = argparse.ArgumentParser()
    args.add_argument(
        "-data", "--dataset", default="NELL-One", type=str,choices=['NELL-One','COVID19-One']
    )
    args.add_argument(
        "-path", "--data_path", default="./NELL", type=str
    )
    args.add_argument(
        "-form", "--data_form", default="Pre-Train", type=str,choices=["Pre-Train", "In-Train", "Discard"])
    args.add_argument("-seed", "--seed", default=None, type=int)
    args.add_argument("-few", "--few", default=1, type=int)
    args.add_argument("-nq", "--num_query", default=3, type=int)
    args.add_argument(
        "-metric",
        "--metric",
        default="Hits@10",
        choices=["MRR", "Hits@10", "Hits@5", "Hits@1"],
    )

    args.add_argument("-dim", "--embed_dim", default=100, type=int)
    args.add_argument("-bs", "--batch_size", default=64, type=int)
    args.add_argument("-lr", "--learning_rate", default=0.001, type=float)
    args.add_argument("-es_p", "--early_stopping_patience", default=3, type=int)

    args.add_argument("-epo", "--epoch", default=10000, type=int) 
    args.add_argument("-prt_epo", "--print_epoch", default=200, type=int)
    args.add_argument("-eval_epo", "--eval_epoch", default=1000, type=int)
    args.add_argument("-ckpt_epo", "--checkpoint_epoch", default=1000, type=int)
    args.add_argument("-p", "--dropout_p", default=0.5, type=float)

    args.add_argument("-gpu", "--device", default=0, type=int)

    args.add_argument("-prefix", "--prefix", default="exp1", type=str)
    args.add_argument(
        "-step", "--step", default="train", type=str, choices=["train", "test", "dev"]
    )
    args.add_argument("-log_dir", "--log_dir", default="log", type=str)
    args.add_argument("-state_dir", "--state_dir", default="state", type=str)
    args.add_argument("-eval_ckpt", "--eval_ckpt", default=None, type=str)
    args.add_argument("-eval_by_rel", "--eval_by_rel", action="store_true")
    args.add_argument("-cpu", "--disable_cuda", action="store_true", default=False)
    args.add_argument("-mask", "--mask_ratio", default=0, type=int)
    args.add_argument("-ft", "--fine_tune", action="store_true")
    args.add_argument("-rum", dest="rum", action="store_true")
    args.add_argument("-no-rum", dest="rum", action="store_false")
    args.add_argument("-vbm", dest="vbm", action="store_true")
    args.add_argument("-no-vbm", dest="vbm", action="store_false")
    args.add_argument(
        "-aggregator",
        "--aggregator",
        default="max",
        choices=["max", "mean", "attn"],
        type=str,
    )
    args.set_defaults(rum=True, vbm=True)
    args = args.parse_args()
    params = {}
    for k, v in vars(args).items():
        params[k] = v
    if not args.disable_cuda and torch.cuda.is_available():
        params["device"] = torch.device("cuda:" + str(args.device))
    else:
        params["device"] = torch.device("cpu")
    if params["data_form"] not in ["Pre-Train","In-train"]:
        # RUM is off by default without pre-trained entity embeddings
        params["rum"] = False

    return params


data_dir = {
    "train_tasks_in_train": "/train_tasks_in_train.json",
    "train_tasks": "/train_tasks.json",
    "test_tasks": "/test_tasks.json",
    "dev_tasks": "/dev_tasks.json",
    "rel2candidates_in_train": "/rel2candidates_in_train.json",
    "rel2candidates": "/rel2candidates.json",
    "e1rel_e2_in_train": "/e1rel_e2_in_train.json",
    "e1rel_e2": "/e1rel_e2.json",
    "ent2ids": "/ent2ids",
    "ent2vec": "/ent2vec.npy",
}
