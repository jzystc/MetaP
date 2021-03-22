
from torch.nn import parameter
from models import *
from tensorboardX import SummaryWriter
import os
import sys
import torch
import shutil
import logging
class Trainer:
    def __init__(self, data_loaders, dataset, parameter):
        self.parameter = parameter
        # data loader
        self.train_data_loader = data_loaders[0]
        self.dev_data_loader = data_loaders[1]
        self.test_data_loader = data_loaders[2]
        # parameters
        self.few = parameter["few"]
        self.num_query = parameter["num_query"]
        self.batch_size = parameter["batch_size"]
        self.learning_rate = parameter["learning_rate"]
        self.early_stopping_patience = parameter["early_stopping_patience"]
        # epoch
        self.epoch = parameter["epoch"]
        self.print_epoch = parameter["print_epoch"]
        self.eval_epoch = parameter["eval_epoch"]
        self.checkpoint_epoch = parameter["checkpoint_epoch"]
        # device
        self.device = parameter["device"]
        self.metaP = MetaP(dataset, parameter)
        self.metaP.to(self.device)
        # optimizer
        self.optimizer = torch.optim.Adam(self.metaP.parameters(), self.learning_rate)
        # tensorboard log writer
        if parameter["step"] == "train":
            self.writer = SummaryWriter(
                os.path.join(parameter["log_dir"] + "_tfevents", parameter["prefix"])
            )
        # dir
        self.state_dir = os.path.join(
            self.parameter["state_dir"], self.parameter["prefix"]
        )
        if not os.path.isdir(self.state_dir):
            os.makedirs(self.state_dir)
        self.ckpt_dir = os.path.join(
            self.parameter["state_dir"], self.parameter["prefix"], "checkpoint"
        )
        if not os.path.isdir(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        self.state_dict_file = ""

        # logging
        if not os.path.isdir(parameter["log_dir"]):
            os.makedirs(parameter["log_dir"])
        logging_dir = os.path.join(
            self.parameter["log_dir"], self.parameter["prefix"] + ".log"
        )
        logging.basicConfig(
            filename=logging_dir, level=logging.INFO, format="%(asctime)s - %(message)s"
        )

        # load state_dict and params
        if parameter["step"] in ["test", "dev"]:
            self.reload()
        self.softmax = nn.Softmax(dim=-1)

    def reload(self):
        if self.parameter["eval_ckpt"] is not None:
            state_dict_file = os.path.join(
                self.ckpt_dir, "state_dict_" + self.parameter["eval_ckpt"] + ".ckpt"
            )
        else:
            state_dict_file = os.path.join(self.state_dir, "state_dict")
        self.state_dict_file = state_dict_file
        logging.info("Reload state_dict from {}".format(state_dict_file))
        print("reload state_dict from {}".format(state_dict_file))
        state = torch.load(state_dict_file, map_location=self.device)
        if os.path.isfile(state_dict_file):
            self.metaP.load_state_dict(state)
        else:
            raise RuntimeError("No state dict in {}!".format(state_dict_file))

    def save_checkpoint(self, epoch):
        torch.save(
            self.metaP.state_dict(),
            os.path.join(self.ckpt_dir, "state_dict_" + str(epoch) + ".ckpt"),
        )

    def del_checkpoint(self, epoch):
        path = os.path.join(self.ckpt_dir, "state_dict_" + str(epoch) + ".ckpt")
        if os.path.exists(path):
            os.remove(path)
        else:
            raise RuntimeError("No such checkpoint to delete: {}".format(path))

    def save_best_state_dict(self, best_epoch):
        shutil.copy(
            os.path.join(self.ckpt_dir, "state_dict_" + str(best_epoch) + ".ckpt"),
            os.path.join(self.state_dir, "state_dict"),
        )

    def write_training_log(self, data, epoch):
        self.writer.add_scalar("Training_Loss", data["Loss"], epoch)

    def write_validating_log(self, data, epoch):
        self.writer.add_scalar("Validating_MRR", data["MRR"], epoch)
        self.writer.add_scalar("Validating_Hits_10", data["Hits@10"], epoch)
        self.writer.add_scalar("Validating_Hits_5", data["Hits@5"], epoch)
        self.writer.add_scalar("Validating_Hits_1", data["Hits@1"], epoch)
        self.writer.add_scalar("Validating_Loss", data["Loss"], epoch)

    def logging_training_data(self, data, epoch):
        logging.info(
            "Epoch: {}\tMRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\tLoss: {:.3f}\r".format(
                epoch,
                data["MRR"],
                data["Hits@10"],
                data["Hits@5"],
                data["Hits@1"],
                data["Loss"],
            )
        )

    def logging_eval_data(self, data, state_path, istest=False):
        setname = "dev set"
        if istest:
            setname = "test set"
        logging.info("Eval {} on {}".format(state_path, setname))
        logging.info(
            "MRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\tLoss: {:.3f}\r".format(
                data["MRR"],
                data["Hits@10"],
                data["Hits@5"],
                data["Hits@1"],
                data["Loss"],
            )
        )

    def rank_predict(self, data, x, ranks):
        # query_idx is the idx of positive score
        query_idx = x.shape[0] - 1
        # sort all scores with descending, because more plausible triplet has higher score
        x, idx = torch.sort(x, descending=True)
        rank = list(idx.cpu().numpy()).index(query_idx) + 1
        ranks.append(rank)
        # update data
        if rank <= 10:
            data["Hits@10"] += 1
        if rank <= 5:
            data["Hits@5"] += 1
        if rank == 1:
            data["Hits@1"] += 1
        data["MRR"] += 1.0 / rank
        return x.cpu().numpy(), rank

    def do_one_step(self, task, iseval=False, curr_rel=""):
        loss, p_score, n_score = 0, 0, 0
        if not iseval:
            self.optimizer.zero_grad()
            score, y_query, p_score, n_score = self.metaP(task, iseval, curr_rel)
            loss = self.metaP.criterion(score, y_query)
            loss.backward()
            self.optimizer.step()
        elif curr_rel != "":
            score, y_query, p_score, n_score = self.metaP(task, iseval, curr_rel)
            loss = self.metaP.criterion(score, y_query)
        return loss, p_score, n_score

    def train(self):
        # initialization
        best_epoch = 0
        best_value = 0
        bad_counts = 0

        # training by epoch
        for e in range(self.epoch):
            # sample one batch from data_loader
            train_task, curr_rel = self.train_data_loader.next_batch()
            loss, p_score, n_score = self.do_one_step(
                train_task, iseval=False, curr_rel=curr_rel
            )
            # print the loss on specific epoch
            if e % self.print_epoch == 0:
                loss_num = loss.item()
                self.write_training_log({"Loss": loss_num}, e)
                print("Epoch: {}\tLoss: {:.4f}".format(e, loss_num))
            # save checkpoint on specific epoch
            if e % self.checkpoint_epoch == 0 and e != 0:
                print("Epoch  {} has finished, saving...".format(e))
                self.save_checkpoint(e)
            # do evaluation on specific epoch
            if e % self.eval_epoch == 0 and e != 0:
                print("Epoch  {} has finished, validating...".format(e))
                valid_data = self.eval(istest=False, epoch=e)
                self.write_validating_log(valid_data, e)
                metric = self.parameter["metric"]
                # early stopping checking
                if valid_data[metric] > best_value:
                    best_value = valid_data[metric]
                    best_epoch = e
                    print(
                        "\tBest model | {0} of valid set is {1:.3f}".format(
                            metric, best_value
                        )
                    )
                    bad_counts = 0
                    # save current best
                    self.save_checkpoint(best_epoch)
                else:
                    print(
                        "\tBest {0} of valid set is {1:.3f} at {2} | bad count is {3}".format(
                            metric, best_value, best_epoch, bad_counts
                        )
                    )
                    bad_counts += 1

                if bad_counts >= self.early_stopping_patience:
                    print("\tEarly stopping at epoch %d" % e)
                    break

        print("Training has finished")
        print(
            "\tBest epoch is {0} | {1} of valid set is {2:.3f}".format(
                best_epoch, metric, best_value
            )
        )
        self.save_best_state_dict(best_epoch)
        print("Finish")

    def eval(self, istest=False, epoch=None):
        self.metaP.eval()
        # clear sharing rel_q
        self.metaP.rel_q_sharing = dict()

        if istest:
            data_loader = self.test_data_loader
        else:
            data_loader = self.dev_data_loader
        data_loader.curr_tri_idx = 0

        # initial return data of validation
        data = {"MRR": 0, "Hits@1": 0, "Hits@5": 0, "Hits@10": 0, "Loss": 0}
        ranks = []

        t = 0
        temp = dict()
        avg_loss = 0
        avg_acc = 0
        while True:
            # sample all the eval tasks
            eval_task, curr_rel = data_loader.next_one_on_eval()
            # print(curr_rel)
            # at the end of sample tasks, a symbol 'EOT' will return
            if eval_task == "EOT":
                break
            t += 1
            loss, p_score, n_score = self.do_one_step(
                eval_task, iseval=True, curr_rel=curr_rel
            )
            x = torch.cat((n_score, p_score), dim=-1).squeeze()
            x = x.detach()
            self.rank_predict(data, x, ranks)
            avg_loss += loss.item()
            # print current temp data dynamically
            for k in data.keys():
                temp[k] = data[k] / t
            sys.stdout.write(
                "{}\tMRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\tLoss: {:.3f}\r".format(
                    t,
                    temp["MRR"],
                    temp["Hits@10"],
                    temp["Hits@5"],
                    temp["Hits@1"],
                    avg_loss / t,
                )
            )
            sys.stdout.flush()

        # print overall evaluation result and return it
        for k in data.keys():
            data[k] = round(data[k] / t, 3)

        if self.parameter["step"] == "train":
            self.logging_training_data(data, epoch)
        else:
            self.logging_eval_data(data, self.state_dict_file, istest)

        print(
            "{}\tMRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\r".format(
                t, data["MRR"], data["Hits@10"], data["Hits@5"], data["Hits@1"]
            )
        )
        # self.metaP.train()
        return data

    def eval_by_relation(self, istest=False, epoch=None):

        self.metaP.eval()
        self.metaP.rel_q_sharing = dict()

        if istest:
            data_loader = self.test_data_loader
        else:
            data_loader = self.dev_data_loader
        data_loader.curr_tri_idx = 0

        all_data = {"MRR": 0, "Hits@1": 0, "Hits@5": 0, "Hits@10": 0}
        all_t = 0
        all_ranks = []
        for rel in data_loader.all_rels:
            
            mean_min = []
            mean_max = []
            mean_rank = []
            mean_num_candidates = []
            mean_score = []
            print(
                "rel: {}, num_cands: {}, num_tasks:{}".format(
                    rel,
                    len(data_loader.rel2candidates[rel]),
                    len(data_loader.tasks[rel][self.few :]),
                )
            )
            data = {"MRR": 0, "Hits@1": 0, "Hits@5": 0, "Hits@10": 0}
            temp = dict()
            t = 0
            ranks = []
            while True:
                eval_task, curr_rel = data_loader.next_one_on_eval_by_relation(rel)
                if eval_task == "EOT":
                    break
                t += 1

                loss, p_score, n_score = self.do_one_step(
                    eval_task, iseval=True, curr_rel=rel
                )
                x = torch.cat([n_score, p_score], 1).squeeze()
                # x = torch.cat([p_score, n_score], 1).squeeze()
                x = x.detach()
                scores, rank = self.rank_predict(data, x, ranks)
                mean_max.append(scores[0])
                mean_min.append(scores[-1])
                mean_rank.append(rank)
                mean_score.append(scores[rank - 1])
                mean_num_candidates.append(len(scores))
                # plt_util.draw_num_entity_to_score(x)
                for k in data.keys():
                    temp[k] = data[k] / t
                sys.stdout.write(
                    "{}\tMRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\r".format(
                        t, temp["MRR"], temp["Hits@10"], temp["Hits@5"], temp["Hits@1"]
                    )
                )
                sys.stdout.flush()

            print(
                "{}\tMRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\r".format(
                    t, temp["MRR"], temp["Hits@10"], temp["Hits@5"], temp["Hits@1"]
                )
            )
            print(
                "mean_min: {:.2f}, mean_max: {:.2f}, mean_score: {:.2f}, mean_rank: {:.2f}, mean_num_candidates: {:.2f}, mrr: {:.3f}".format(
                    np.mean(mean_min),
                    np.mean(mean_max),
                    np.mean(mean_score),
                    np.mean(mean_rank),
                    np.mean(mean_num_candidates),
                    temp["MRR"]
                )
            )
            for k in data.keys():
                all_data[k] += data[k]
            all_t += t
            all_ranks.extend(ranks)
        print("Overall")
        for k in all_data.keys():
            all_data[k] = round(all_data[k] / all_t, 3)
        print(
            "{}\tMRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\r".format(
                all_t,
                all_data["MRR"],
                all_data["Hits@10"],
                all_data["Hits@5"],
                all_data["Hits@1"],
            )
        )

        return all_data
