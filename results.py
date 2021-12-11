import argparse
import os
import pandas as pd
from matplotlib import pyplot as plt
import tensorboard as tb
import numpy as np

# arg parse
parser = argparse.ArgumentParser(description='re-sln results')
parser.add_argument('--experiment_id', type=str, help='the TensorboardDEV experiment id', required=True)

results_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "results")

experiments = \
{
    "cifar10":
        {
            "ce":
                {
                    "symmetric": 
                        {
                            "paper": "exp_2021-11-25 13:17:26.851200", "custom": "exp_2021-11-25 20:30:28.794160"
                        },
                    "asymmetric":
                        {
                            "paper": "exp_2021-11-26 09:21:19.524188", "custom": "exp_2021-11-26 14:03:18.975684"
                        },
                    "dependent":
                        {
                            "paper": "exp_2021-11-26 20:14:40.983299"
                        },
                    "openset":
                        {
                            "custom": "exp_2021-11-27 13:35:23.026659"
                        },
                },
            "sln":
                {
                    "symmetric":
                        {
                            "paper": "exp_2021-11-25 15:38:09.361059", "custom": "exp_2021-11-25 20:31:37.546765"
                        },
                    "asymmetric":
                        {
                            "paper": "exp_2021-11-26 11:41:56.758060", "custom": "exp_2021-11-26 16:11:00.844488"
                        },
                    "dependent":
                        {
                            "paper": "exp_2021-11-27 11:07:55.847340"
                        },
                    "openset":
                        {
                            "custom": "exp_2021-11-27 13:44:37.885816"
                        },
                },
            "sln-mo":
                {
                    "symmetric":
                        {
                            "paper": "exp_2021-11-25 16:46:29.066838", "custom": "exp_2021-11-26 09:18:14.291265"
                        },
                    "asymmetric":
                        {
                            "paper": "exp_2021-11-26 11:44:27.727904", "custom": "exp_2021-11-26 16:14:06.628600"
                        },
                    "dependent":
                        {
                            "paper": "exp_2021-11-27 11:11:07.020347"
                        },
                    "openset":
                        {
                            "custom": "exp_2021-11-27 13:46:43.777573"
                        },
                },
            "sln-mo-lc":
                {
                    "symmetric":
                        {
                            "paper": "exp_2021-11-26 11:18:36.051172", "custom": "exp_2021-11-26 13:51:03.590616"
                        },
                    "asymmetric":
                        {
                            "paper": "exp_2021-11-26 13:57:45.567433", "custom": "exp_2021-11-26 16:16:06.031597"
                        },
                    "dependent":
                        {
                            "paper": "exp_2021-11-27 11:14:24.120092"
                        },
                    "openset":
                        {
                            "custom": "exp_2021-11-28 16:34:38.935269"
                        },
                },
        },
    "cifar100":
        {
            "ce":
                {
                    "symmetric":
                        {
                            "paper": "exp_2021-11-29 13:02:42.947124", "custom": "exp_2021-11-29 15:14:24.277293"
                        },
                    "asymmetric":
                        {
                            "paper": "exp_2021-12-02 17:15:05.141925", "custom": "exp_2021-12-02 20:50:30.272408"
                        },
                    "dependent":
                        {
                            "paper": "exp_2021-12-03 12:09:50.374569"
                        },
                },
            "sln":
                {
                    "symmetric":
                        {
                            "paper": "exp_2021-11-29 13:12:28.474547", "custom": "exp_2021-11-29 15:15:36.143703"
                        },
                    "asymmetric":
                        {
                            "paper": "exp_2021-12-02 17:34:08.440889", "custom": "exp_2021-12-02 20:55:53.387841"
                        },
                    "dependent":
                        {
                            "paper": "exp_2021-12-03 14:37:51.783033"
                        },
                },
            "sln-mo":
                {
                    "symmetric":
                        {
                            "paper": "exp_2021-11-29 13:16:11.590910", "custom": "exp_2021-11-29 22:15:08.652843"
                        },
                    "asymmetric":
                        {
                            "paper": "exp_2021-12-02 17:39:34.952358", "custom": "exp_2021-12-03 11:53:37.290785"
                        },
                    "dependent":
                        {
                            "paper": "exp_2021-12-03 14:43:27.237441"
                        },
                },
            "sln-mo-lc":
                {
                    "symmetric":
                        {
                            "paper": "exp_2021-11-29 22:04:19.910053", "custom": "exp_2021-11-29 22:26:18.532929"
                        },
                    "asymmetric":
                        {
                            "paper": "exp_2021-12-02 20:43:32.204172", "custom": "exp_2021-12-03 12:01:04.662910"
                        },
                    "dependent":
                        {
                            "paper": "exp_2021-12-03 14:51:11.441549"
                        },
                },
        },
    "animal-10n":
        {
            "ce":
                {
                    "real":
                        {
                            "run1": "exp_2021-12-08 11:38:16.477097", "run2": "exp_2021-12-08 20:23:04.646474"
                        }
                },
            "sln":
                {
                    "real":
                        {
                            "run1": "exp_2021-12-08 13:06:15.220761", "run2": "exp_2021-12-08 18:29:16.424429"
                        }
                },
            "sln-mo":
                {
                    "real":
                        {
                            "run1": "exp_2021-12-08 14:43:36.523374", "run2": "exp_2021-12-08 18:29:27.093767"
                        }
                },
            "sln-mo-lc":
                {
                    "real":
                        {
                            "run1": "exp_2021-12-07 21:55:00.730335", "run2": "exp_2021-12-08 18:35:55.422639"
                        }
                }
        }
}

tags = \
{
    "loss":
        {
            "train": "loss/train", "test": "loss/test"
        },
    "accuracy":
        {
            "train": "accuracy/train", "test": "accuracy/test"
        }
}

suffixes = \
{
    "loss":
        {
            "train":
                {
                    "all": "/loss_train_all", "clean": "/loss_train_clean", "noisy": "/loss_train_noisy"
                }
        }
}


def get_accuracy(dataset_name, model, noise_mode, if_train):
    if_train_str = "train" if if_train else "test"
    
    if len(experiments[dataset_name][model][noise_mode]) == 2:
        e1 = \
            np.array(df[np.logical_and(
                df["run"] == experiments[dataset_name][model][noise_mode]["paper"], 
                df["tag"] == tags["accuracy"][if_train_str])]["value"])
        e2 = \
            np.array(df[np.logical_and(
                df["run"] == experiments[dataset_name][model][noise_mode]["custom"], 
                df["tag"] == tags["accuracy"][if_train_str])]["value"])
        e = (e1 + e2) / 2
        mean = e[-1]
        std = np.sqrt(((mean - e1[-1])**2 + (mean - e2[-1])**2)/2)
    else:
        key = list(experiments[dataset_name][model][noise_mode].keys())[0]
        e = \
            np.array(df[np.logical_and(
                df["run"] == experiments[dataset_name][model][noise_mode][key], 
                df["tag"] == tags["accuracy"][if_train_str])]["value"])
        mean = e[-1]
        std = 0
    
    return e, mean, std


def get_accuracy_animal(model, if_train):
    if_train_str = "train" if if_train else "test"
    
    
    e1 = \
        np.array(df[np.logical_and(
            df["run"] == experiments["animal-10n"][model]["real"]["run1"], 
            df["tag"] == tags["accuracy"][if_train_str])]["value"])
    e2 = \
        np.array(df[np.logical_and(
            df["run"] == experiments["animal-10n"][model]["real"]["run2"], 
            df["tag"] == tags["accuracy"][if_train_str])]["value"])
    e = (e1 + e2) / 2
    mean = e[-1]
    std = np.sqrt(((mean - e1[-1])**2 + (mean - e2[-1])**2)/2)
    
    return e, mean, std


def get_loss_train(dataset_name, model, noise_mode, loss_type):
    
    if len(experiments[dataset_name][model][noise_mode]) == 2:
        e1 = \
            np.array(df[np.logical_and(
                df["run"] == experiments[dataset_name][model][noise_mode]["paper"] + suffixes["loss"]["train"][loss_type], 
                df["tag"] == tags["loss"]["train"])]["value"])
        e2 = \
            np.array(df[np.logical_and(
                df["run"] == experiments[dataset_name][model][noise_mode]["custom"] + suffixes["loss"]["train"][loss_type], 
                df["tag"] == tags["loss"]["train"])]["value"])
        e = (e1 + e2) / 2
    else:
        key = list(experiments[dataset_name][model][noise_mode].keys())[0]
        e = \
            np.array(df[np.logical_and(
                df["run"] == experiments[dataset_name][model][noise_mode][key] + suffixes["loss"]["train"][loss_type], 
                df["tag"] == tags["loss"]["train"])]["value"])
    return e

def get_loss_test(dataset_name, model, noise_mode):
    if_train_str = "train" if if_train else "test"
    
    if len(experiments[dataset_name][model][noise_mode]) == 2:
        e1 = \
            np.array(df[np.logical_and(
                df["run"] == experiments[dataset_name][model][noise_mode]["paper"], 
                df["tag"] == tags["loss"]["test"])]["value"])
        e2 = \
            np.array(df[np.logical_and(
                df["run"] == experiments[dataset_name][model][noise_mode]["custom"], 
                df["tag"] == tags["loss"]["test"])]["value"])
        e = (e1 + e2) / 2
    else:
        key = list(experiments[dataset_name][model][noise_mode].keys())[0]
        e = \
            np.array(df[np.logical_and(
                df["run"] == experiments[dataset_name][model][noise_mode][key], 
                df["tag"] == tags["loss"]["test"])]["value"])
    
    return e


if __name__ == "__main__":
    # args parse
    args = parser.parse_args()
    
    experiment = tb.data.experimental.ExperimentFromDev(args.experiment_id)
    df = experiment.get_scalars(pivot=False)
    print("retrieved data from TensorboardDEV")
    
    # c10 ce test acc
    dataset_name, model, noise_mode, if_train = "cifar10", "ce", "symmetric", False
    c10_ce_sym_acc, c10_ce_sym_acc_mean, c10_ce_sym_acc_std = get_accuracy(dataset_name, model, noise_mode, if_train)

    dataset_name, model, noise_mode, if_train = "cifar10", "ce", "asymmetric", False
    c10_ce_asym_acc, c10_ce_asym_acc_mean, c10_ce_asym_acc_std = get_accuracy(dataset_name, model, noise_mode, if_train)

    dataset_name, model, noise_mode, if_train = "cifar10", "ce", "dependent", False
    c10_ce_dep_acc, c10_ce_dep_acc_mean, c10_ce_dep_acc_std = get_accuracy(dataset_name, model, noise_mode, if_train)

    dataset_name, model, noise_mode, if_train = "cifar10", "ce", "openset", False
    c10_ce_openset_acc, c10_ce_openset_acc_mean, c10_ce_openset_acc_std = get_accuracy(dataset_name, model, noise_mode, if_train)

    plt.figure(figsize=(6, 4))
    plt.plot(c10_ce_sym_acc*100, color="blue", label="sym")
    plt.plot(c10_ce_asym_acc*100, color="orange", label="asym")
    plt.plot(c10_ce_dep_acc*100, color="green", label="dep")
    plt.plot(c10_ce_openset_acc*100, color="red", label="openset")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Test Accuracy (CE) on CIFAR-10")
    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    plt.xlim(0, 300)
    plt.ylim(50, 87)
    plt.tight_layout()
    fig_name = "c10-ce-test-acc.png"
    plt.savefig(os.path.join(results_path, fig_name))
    
    # c10 sln test acc
    dataset_name, model, noise_mode, if_train = "cifar10", "sln", "symmetric", False
    c10_sln_sym_acc, c10_sln_sym_acc_mean, c10_sln_sym_acc_std = get_accuracy(dataset_name, model, noise_mode, if_train)

    dataset_name, model, noise_mode, if_train = "cifar10", "sln", "asymmetric", False
    c10_sln_asym_acc, c10_sln_asym_acc_mean, c10_sln_asym_acc_std = get_accuracy(dataset_name, model, noise_mode, if_train)

    dataset_name, model, noise_mode, if_train = "cifar10", "sln", "dependent", False
    c10_sln_dep_acc, c10_sln_dep_acc_mean, c10_sln_dep_acc_std = get_accuracy(dataset_name, model, noise_mode, if_train)

    dataset_name, model, noise_mode, if_train = "cifar10", "sln", "openset", False
    c10_sln_openset_acc, c10_sln_openset_acc_mean, c10_sln_openset_acc_std = get_accuracy(dataset_name, model, noise_mode, if_train)

    plt.figure(figsize=(6, 4))
    plt.plot(c10_sln_sym_acc*100, color="blue", label="sym")
    plt.plot(c10_sln_asym_acc*100, color="orange", label="asym")
    plt.plot(c10_sln_dep_acc*100, color="green", label="dep")
    plt.plot(c10_sln_openset_acc*100, color="red", label="openset")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Test Accuracy (SLN) on CIFAR-10")
    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    plt.xlim(0, 300)
    plt.ylim(50, 87)
    plt.tight_layout()
    fig_name = "c10-sln-test-acc.png"
    plt.savefig(os.path.join(results_path, fig_name))
    
    # c10 sln-mo test acc
    dataset_name, model, noise_mode, if_train = "cifar10", "sln-mo", "symmetric", False
    c10_sln_mo_sym_acc, c10_sln_mo_sym_acc_mean, c10_sln_mo_sym_acc_std = get_accuracy(dataset_name, model, noise_mode, if_train)

    dataset_name, model, noise_mode, if_train = "cifar10", "sln-mo", "asymmetric", False
    c10_sln_mo_asym_acc, c10_sln_mo_asym_acc_mean, c10_sln_mo_asym_acc_std = get_accuracy(dataset_name, model, noise_mode, if_train)

    dataset_name, model, noise_mode, if_train = "cifar10", "sln-mo", "dependent", False
    c10_sln_mo_dep_acc, c10_sln_mo_dep_acc_mean, c10_sln_mo_dep_acc_std = get_accuracy(dataset_name, model, noise_mode, if_train)

    dataset_name, model, noise_mode, if_train = "cifar10", "sln-mo", "openset", False
    c10_sln_mo_openset_acc, c10_sln_mo_openset_acc_mean, c10_sln_mo_openset_acc_std = get_accuracy(dataset_name, model, noise_mode, if_train)

    plt.figure(figsize=(6, 4))
    plt.plot(c10_sln_mo_sym_acc*100, color="blue", label="sym")
    plt.plot(c10_sln_mo_asym_acc*100, color="orange", label="asym")
    plt.plot(c10_sln_mo_dep_acc*100, color="green", label="dep")
    plt.plot(c10_sln_mo_openset_acc*100, color="red", label="openset")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Test Accuracy (SLN-MO) on CIFAR-10")
    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    plt.xlim(0, 300)
    plt.ylim(50, 92)
    plt.tight_layout()
    fig_name = "c10-sln-mo-test-acc.png"
    plt.savefig(os.path.join(results_path, fig_name))
    
    # c10 sln-mo-lc test acc
    dataset_name, model, noise_mode, if_train = "cifar10", "sln-mo-lc", "symmetric", False
    c10_sln_mo_lc_sym_acc, c10_sln_mo_lc_sym_acc_mean, c10_sln_mo_lc_sym_acc_std = get_accuracy(dataset_name, model, noise_mode, if_train)

    dataset_name, model, noise_mode, if_train = "cifar10", "sln-mo-lc", "asymmetric", False
    c10_sln_mo_lc_asym_acc, c10_sln_mo_lc_asym_acc_mean, c10_sln_mo_lc_asym_acc_std = get_accuracy(dataset_name, model, noise_mode, if_train)

    dataset_name, model, noise_mode, if_train = "cifar10", "sln-mo-lc", "dependent", False
    c10_sln_mo_lc_dep_acc, c10_sln_mo_lc_dep_acc_mean, c10_sln_mo_lc_dep_acc_std = get_accuracy(dataset_name, model, noise_mode, if_train)

    dataset_name, model, noise_mode, if_train = "cifar10", "sln-mo-lc", "openset", False
    c10_sln_mo_lc_openset_acc, c10_sln_mo_lc_openset_acc_mean, c10_sln_mo_lc_openset_acc_std = get_accuracy(dataset_name, model, noise_mode, if_train)

    plt.figure(figsize=(6, 4))
    plt.plot(c10_sln_mo_lc_sym_acc*100, color="blue", label="sym")
    plt.plot(c10_sln_mo_lc_asym_acc*100, color="orange", label="asym")
    plt.plot(c10_sln_mo_lc_dep_acc*100, color="green", label="dep")
    plt.plot(c10_sln_mo_lc_openset_acc*100, color="red", label="openset")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Test Accuracy (SLN-MO-LC) on CIFAR-10")
    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    plt.xlim(0, 300)
    plt.ylim(50, 92)
    plt.tight_layout()
    fig_name = "c10-sln-mo-lc-test-acc.png"
    plt.savefig(os.path.join(results_path, fig_name))
    
    plt.figure(figsize=(6, 4))
    plt.plot(c10_sln_mo_lc_sym_acc*100, color="blue", label="sym")
    plt.plot(c10_sln_mo_lc_asym_acc*100, color="orange", label="asym")
    plt.plot(c10_sln_mo_lc_dep_acc*100, color="green", label="dep")
    plt.plot(c10_sln_mo_lc_openset_acc*100, color="red", label="openset")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Test Accuracy (SLN-MO-LC) on CIFAR-10")
    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    plt.xlim(220, 300)
    plt.ylim(80, 92)
    plt.tight_layout()
    fig_name = "c10-sln-mo-lc-test-acc-zoom.png"
    plt.savefig(os.path.join(results_path, fig_name))
    
    # c10 test acc table
    data_cifar10 = {
        "model": ["ce", "sln", "sln-mo", "sln-mo-lc"],
        "symmetric": [
            str(round(c10_ce_sym_acc_mean*100,2))+u"\u00B1"+str(round(c10_ce_sym_acc_std*100,2)),
            str(round(c10_sln_sym_acc_mean*100,2))+u"\u00B1"+str(round(c10_sln_sym_acc_std*100,2)),
            str(round(c10_sln_mo_sym_acc_mean*100,2))+u"\u00B1"+str(round(c10_sln_mo_sym_acc_std*100,2)),
            str(round(c10_sln_mo_lc_sym_acc_mean*100,2))+u"\u00B1"+str(round(c10_sln_mo_lc_sym_acc_std*100,2)),
        ],
        "asymmetric": [
            str(round(c10_ce_asym_acc_mean*100,2))+u"\u00B1"+str(round(c10_ce_asym_acc_std*100,2)),
            str(round(c10_sln_asym_acc_mean*100,2))+u"\u00B1"+str(round(c10_sln_asym_acc_std*100,2)),
            str(round(c10_sln_mo_asym_acc_mean*100,2))+u"\u00B1"+str(round(c10_sln_mo_asym_acc_std*100,2)),
            str(round(c10_sln_mo_lc_asym_acc_mean*100,2))+u"\u00B1"+str(round(c10_sln_mo_lc_asym_acc_std*100,2)),
        ],
        "dependent": [
            str(round(c10_ce_dep_acc_mean*100,2))+u"\u00B1"+str(round(c10_ce_dep_acc_std*100,2)),
            str(round(c10_sln_dep_acc_mean*100,2))+u"\u00B1"+str(round(c10_sln_dep_acc_std*100,2)),
            str(round(c10_sln_mo_dep_acc_mean*100,2))+u"\u00B1"+str(round(c10_sln_mo_dep_acc_std*100,2)),
            str(round(c10_sln_mo_lc_dep_acc_mean*100,2))+u"\u00B1"+str(round(c10_sln_mo_lc_dep_acc_std*100,2)),
        ],
        "openset": [
            str(round(c10_ce_openset_acc_mean*100,2))+u"\u00B1"+str(round(c10_ce_openset_acc_std*100,2)),
            str(round(c10_sln_openset_acc_mean*100,2))+u"\u00B1"+str(round(c10_sln_openset_acc_std*100,2)),
            str(round(c10_sln_mo_openset_acc_mean*100,2))+u"\u00B1"+str(round(c10_sln_mo_openset_acc_std*100,2)),
            str(round(c10_sln_mo_lc_openset_acc_mean*100,2))+u"\u00B1"+str(round(c10_sln_mo_lc_openset_acc_std*100,2)),
        ],
    }
    df_table_cifar10 = pd.DataFrame(data=data_cifar10)
    table_name = "c10-test-acc-table.csv"
    df_table_cifar10.to_csv(os.path.join(results_path, table_name))
    
    print("results have been generated")
    
    # c10 ce training losses

    dataset_name, model, noise_mode, loss_type = "cifar10", "ce", "symmetric", "clean"
    c10_ce_sym_loss_train_clean = get_loss_train(dataset_name, model, noise_mode, loss_type)
    dataset_name, model, noise_mode, loss_type = "cifar10", "ce", "symmetric", "noisy"
    c10_ce_sym_loss_train_noisy = get_loss_train(dataset_name, model, noise_mode, loss_type)

    dataset_name, model, noise_mode, loss_type = "cifar10", "ce", "asymmetric", "clean"
    c10_ce_asym_loss_train_clean = get_loss_train(dataset_name, model, noise_mode, loss_type)
    dataset_name, model, noise_mode, loss_type = "cifar10", "ce", "asymmetric", "noisy"
    c10_ce_asym_loss_train_noisy = get_loss_train(dataset_name, model, noise_mode, loss_type)

    dataset_name, model, noise_mode, loss_type = "cifar10", "ce", "dependent", "clean"
    c10_ce_dep_loss_train_clean = get_loss_train(dataset_name, model, noise_mode, loss_type)
    dataset_name, model, noise_mode, loss_type = "cifar10", "ce", "dependent", "noisy"
    c10_ce_dep_loss_train_noisy = get_loss_train(dataset_name, model, noise_mode, loss_type)

    dataset_name, model, noise_mode, loss_type = "cifar10", "ce", "openset", "clean"
    c10_ce_openset_loss_train_clean = get_loss_train(dataset_name, model, noise_mode, loss_type)
    dataset_name, model, noise_mode, loss_type = "cifar10", "ce", "openset", "noisy"
    c10_ce_openset_loss_train_noisy = get_loss_train(dataset_name, model, noise_mode, loss_type)

    plt.figure(figsize=(6, 4))
    plt.plot(c10_ce_sym_loss_train_clean, color="blue", label="sym clean")
    plt.plot(c10_ce_sym_loss_train_noisy, color="blue", linestyle='dashed', label="sym noisy")
    plt.plot(c10_ce_asym_loss_train_clean, color="orange", label="asym clean")
    plt.plot(c10_ce_asym_loss_train_noisy, color="orange", linestyle='dashed', label="asym noisy")
    plt.plot(c10_ce_dep_loss_train_clean, color="green", label="dep clean")
    plt.plot(c10_ce_dep_loss_train_noisy, color="green", linestyle='dashed', label="dep noisy")
    plt.plot(c10_ce_openset_loss_train_clean, color="red", label="openset clean")
    plt.plot(c10_ce_openset_loss_train_noisy, color="red", linestyle='dashed', label="openset noisy")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CE Training Loss (CE) on CIFAR-10")
    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    plt.xlim(0, 300)
    plt.ylim(0, 3)
    plt.tight_layout()
    fig_name = "c10-ce-train-loss.png"
    plt.savefig(os.path.join(results_path, fig_name))
    
    # c10 sln train losses
    dataset_name, model, noise_mode, loss_type = "cifar10", "sln", "symmetric", "clean"
    c10_sln_sym_loss_train_clean = get_loss_train(dataset_name, model, noise_mode, loss_type)
    dataset_name, model, noise_mode, loss_type = "cifar10", "sln", "symmetric", "noisy"
    c10_sln_sym_loss_train_noisy = get_loss_train(dataset_name, model, noise_mode, loss_type)

    dataset_name, model, noise_mode, loss_type = "cifar10", "sln", "asymmetric", "clean"
    c10_sln_asym_loss_train_clean = get_loss_train(dataset_name, model, noise_mode, loss_type)
    dataset_name, model, noise_mode, loss_type = "cifar10", "sln", "asymmetric", "noisy"
    c10_sln_asym_loss_train_noisy = get_loss_train(dataset_name, model, noise_mode, loss_type)

    dataset_name, model, noise_mode, loss_type = "cifar10", "sln", "dependent", "clean"
    c10_sln_dep_loss_train_clean = get_loss_train(dataset_name, model, noise_mode, loss_type)
    dataset_name, model, noise_mode, loss_type = "cifar10", "sln", "dependent", "noisy"
    c10_sln_dep_loss_train_noisy = get_loss_train(dataset_name, model, noise_mode, loss_type)

    dataset_name, model, noise_mode, loss_type = "cifar10", "sln", "openset", "clean"
    c10_sln_openset_loss_train_clean = get_loss_train(dataset_name, model, noise_mode, loss_type)
    dataset_name, model, noise_mode, loss_type = "cifar10", "sln", "openset", "noisy"
    c10_sln_openset_loss_train_noisy = get_loss_train(dataset_name, model, noise_mode, loss_type)

    plt.figure(figsize=(6, 4))
    plt.plot(c10_sln_sym_loss_train_clean, color="blue", label="sym clean")
    plt.plot(c10_sln_sym_loss_train_noisy, color="blue", linestyle='dashed', label="sym noisy")
    plt.plot(c10_sln_asym_loss_train_clean, color="orange", label="asym clean")
    plt.plot(c10_sln_asym_loss_train_noisy, color="orange", linestyle='dashed', label="asym noisy")
    plt.plot(c10_sln_dep_loss_train_clean, color="green", label="dep clean")
    plt.plot(c10_sln_dep_loss_train_noisy, color="green", linestyle='dashed', label="dep noisy")
    plt.plot(c10_sln_openset_loss_train_clean, color="red", label="openset clean")
    plt.plot(c10_sln_openset_loss_train_noisy, color="red", linestyle='dashed', label="openset noisy")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CE Training Loss (SLN) on CIFAR-10")
    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    plt.xlim(0, 300)
    plt.ylim(0, 3)
    plt.tight_layout()
    fig_name = "c10-sln-train-loss.png"
    plt.savefig(os.path.join(results_path, fig_name))
    
    # c10 sln-mo train losses
    dataset_name, model, noise_mode, loss_type = "cifar10", "sln-mo", "symmetric", "clean"
    c10_sln_mo_sym_loss_train_clean = get_loss_train(dataset_name, model, noise_mode, loss_type)
    dataset_name, model, noise_mode, loss_type = "cifar10", "sln-mo", "symmetric", "noisy"
    c10_sln_mo_sym_loss_train_noisy = get_loss_train(dataset_name, model, noise_mode, loss_type)

    dataset_name, model, noise_mode, loss_type = "cifar10", "sln-mo", "asymmetric", "clean"
    c10_sln_mo_asym_loss_train_clean = get_loss_train(dataset_name, model, noise_mode, loss_type)
    dataset_name, model, noise_mode, loss_type = "cifar10", "sln-mo", "asymmetric", "noisy"
    c10_sln_mo_asym_loss_train_noisy = get_loss_train(dataset_name, model, noise_mode, loss_type)

    dataset_name, model, noise_mode, loss_type = "cifar10", "sln-mo", "dependent", "clean"
    c10_sln_mo_dep_loss_train_clean = get_loss_train(dataset_name, model, noise_mode, loss_type)
    dataset_name, model, noise_mode, loss_type = "cifar10", "sln-mo", "dependent", "noisy"
    c10_sln_mo_dep_loss_train_noisy = get_loss_train(dataset_name, model, noise_mode, loss_type)

    dataset_name, model, noise_mode, loss_type = "cifar10", "sln-mo", "openset", "clean"
    c10_sln_mo_openset_loss_train_clean = get_loss_train(dataset_name, model, noise_mode, loss_type)
    dataset_name, model, noise_mode, loss_type = "cifar10", "sln-mo", "openset", "noisy"
    c10_sln_mo_openset_loss_train_noisy = get_loss_train(dataset_name, model, noise_mode, loss_type)

    plt.figure(figsize=(6, 4))
    plt.plot(c10_sln_mo_sym_loss_train_clean, color="blue", label="sym clean")
    plt.plot(c10_sln_mo_sym_loss_train_noisy, color="blue", linestyle='dashed', label="sym noisy")
    plt.plot(c10_sln_mo_asym_loss_train_clean, color="orange", label="asym clean")
    plt.plot(c10_sln_mo_asym_loss_train_noisy, color="orange", linestyle='dashed', label="asym noisy")
    plt.plot(c10_sln_mo_dep_loss_train_clean, color="green", label="dep clean")
    plt.plot(c10_sln_mo_dep_loss_train_noisy, color="green", linestyle='dashed', label="dep noisy")
    plt.plot(c10_sln_mo_openset_loss_train_clean, color="red", label="openset clean")
    plt.plot(c10_sln_mo_openset_loss_train_noisy, color="red", linestyle='dashed', label="openset noisy")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CE Training Loss (SLN-MO) on CIFAR-10")
    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    plt.xlim(0, 300)
    plt.ylim(0, 3)
    plt.tight_layout()
    fig_name = "c10-sln-mo-train-loss.png"
    plt.savefig(os.path.join(results_path, fig_name))
    
    # c10 sln-mo-lc train losses

    dataset_name, model, noise_mode, loss_type = "cifar10", "sln-mo-lc", "symmetric", "clean"
    c10_sln_mo_lc_sym_loss_train_clean = get_loss_train(dataset_name, model, noise_mode, loss_type)
    dataset_name, model, noise_mode, loss_type = "cifar10", "sln-mo-lc", "symmetric", "noisy"
    c10_sln_mo_lc_sym_loss_train_noisy = get_loss_train(dataset_name, model, noise_mode, loss_type)

    dataset_name, model, noise_mode, loss_type = "cifar10", "sln-mo-lc", "asymmetric", "clean"
    c10_sln_mo_lc_asym_loss_train_clean = get_loss_train(dataset_name, model, noise_mode, loss_type)
    dataset_name, model, noise_mode, loss_type = "cifar10", "sln-mo-lc", "asymmetric", "noisy"
    c10_sln_mo_lc_asym_loss_train_noisy = get_loss_train(dataset_name, model, noise_mode, loss_type)

    dataset_name, model, noise_mode, loss_type = "cifar10", "sln-mo-lc", "dependent", "clean"
    c10_sln_mo_lc_dep_loss_train_clean = get_loss_train(dataset_name, model, noise_mode, loss_type)
    dataset_name, model, noise_mode, loss_type = "cifar10", "sln-mo-lc", "dependent", "noisy"
    c10_sln_mo_lc_dep_loss_train_noisy = get_loss_train(dataset_name, model, noise_mode, loss_type)

    dataset_name, model, noise_mode, loss_type = "cifar10", "sln-mo-lc", "openset", "clean"
    c10_sln_mo_lc_openset_loss_train_clean = get_loss_train(dataset_name, model, noise_mode, loss_type)
    dataset_name, model, noise_mode, loss_type = "cifar10", "sln-mo-lc", "openset", "noisy"
    c10_sln_mo_lc_openset_loss_train_noisy = get_loss_train(dataset_name, model, noise_mode, loss_type)

    plt.figure(figsize=(6, 4))
    plt.plot(c10_sln_mo_lc_sym_loss_train_clean, color="blue", label="sym clean")
    plt.plot(c10_sln_mo_lc_sym_loss_train_noisy, color="blue", linestyle='dashed', label="sym noisy")
    plt.plot(c10_sln_mo_lc_asym_loss_train_clean, color="orange", label="asym clean")
    plt.plot(c10_sln_mo_lc_asym_loss_train_noisy, color="orange", linestyle='dashed', label="asym noisy")
    plt.plot(c10_sln_mo_lc_dep_loss_train_clean, color="green", label="dep clean")
    plt.plot(c10_sln_mo_lc_dep_loss_train_noisy, color="green", linestyle='dashed', label="dep noisy")
    plt.plot(c10_sln_mo_lc_openset_loss_train_clean, color="red", label="openset clean")
    plt.plot(c10_sln_mo_lc_openset_loss_train_noisy, color="red", linestyle='dashed', label="openset noisy")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CE Training Loss (SLN-MO-LC) on CIFAR-10")
    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    plt.xlim(0, 300)
    plt.ylim(0, 3)
    plt.tight_layout()
    fig_name = "c10-sln-mo-lc-train-loss.png"
    plt.savefig(os.path.join(results_path, fig_name))
    
    # cifar100
    # c100 ce test acc
    dataset_name, model, noise_mode, if_train = "cifar100", "ce", "symmetric", False
    c100_ce_sym_acc, c100_ce_sym_acc_mean, c100_ce_sym_acc_std = get_accuracy(dataset_name, model, noise_mode, if_train)

    dataset_name, model, noise_mode, if_train = "cifar100", "ce", "asymmetric", False
    c100_ce_asym_acc, c100_ce_asym_acc_mean, c100_ce_asym_acc_std = get_accuracy(dataset_name, model, noise_mode, if_train)

    dataset_name, model, noise_mode, if_train = "cifar100", "ce", "dependent", False
    c100_ce_dep_acc, c100_ce_dep_acc_mean, c100_ce_dep_acc_std = get_accuracy(dataset_name, model, noise_mode, if_train)

    plt.figure(figsize=(6, 4))
    plt.plot(c100_ce_sym_acc*100, color="blue", label="sym")
    plt.plot(c100_ce_asym_acc*100, color="orange", label="asym")
    plt.plot(c100_ce_dep_acc*100, color="green", label="dep")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Test Accuracy (CE) on CIFAR-100")
    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    plt.xlim(0, 300)
    plt.ylim(10, 72)
    plt.tight_layout()
    fig_name = "c100-ce-test-acc.png"
    plt.savefig(os.path.join(results_path, fig_name))
    
    # c100 sln test acc
    dataset_name, model, noise_mode, if_train = "cifar100", "sln", "symmetric", False
    c100_sln_sym_acc, c100_sln_sym_acc_mean, c100_sln_sym_acc_std = get_accuracy(dataset_name, model, noise_mode, if_train)

    dataset_name, model, noise_mode, if_train = "cifar100", "sln", "asymmetric", False
    c100_sln_asym_acc, c100_sln_asym_acc_mean, c100_sln_asym_acc_std = get_accuracy(dataset_name, model, noise_mode, if_train)

    dataset_name, model, noise_mode, if_train = "cifar100", "sln", "dependent", False
    c100_sln_dep_acc, c100_sln_dep_acc_mean, c100_sln_dep_acc_std = get_accuracy(dataset_name, model, noise_mode, if_train)

    plt.figure(figsize=(6, 4))
    plt.plot(c100_sln_sym_acc*100, color="blue", label="sym")
    plt.plot(c100_sln_asym_acc*100, color="orange", label="asym")
    plt.plot(c100_sln_dep_acc*100, color="green", label="dep")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Test Accuracy (SLN) on CIFAR-100")
    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    plt.xlim(0, 300)
    plt.ylim(10, 72)
    plt.tight_layout()
    fig_name = "c100-sln-test-acc.png"
    plt.savefig(os.path.join(results_path, fig_name))
    
    # c100 sln-mo test acc
    dataset_name, model, noise_mode, if_train = "cifar100", "sln-mo", "symmetric", False
    c100_sln_mo_sym_acc, c100_sln_mo_sym_acc_mean, c100_sln_mo_sym_acc_std = get_accuracy(dataset_name, model, noise_mode, if_train)

    dataset_name, model, noise_mode, if_train = "cifar100", "sln-mo", "asymmetric", False
    c100_sln_mo_asym_acc, c100_sln_mo_asym_acc_mean, c100_sln_mo_asym_acc_std = get_accuracy(dataset_name, model, noise_mode, if_train)

    dataset_name, model, noise_mode, if_train = "cifar100", "sln-mo", "dependent", False
    c100_sln_mo_dep_acc, c100_sln_mo_dep_acc_mean, c100_sln_mo_dep_acc_std = get_accuracy(dataset_name, model, noise_mode, if_train)

    plt.figure(figsize=(6, 4))
    plt.plot(c100_sln_mo_sym_acc*100, color="blue", label="sym")
    plt.plot(c100_sln_mo_asym_acc*100, color="orange", label="asym")
    plt.plot(c100_sln_mo_dep_acc*100, color="green", label="dep")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Test Accuracy (SLN-MO) on CIFAR-100")
    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    plt.xlim(0, 300)
    plt.ylim(10, 72)
    plt.tight_layout()
    fig_name = "c100-sln-mo-test-acc.png"
    plt.savefig(os.path.join(results_path, fig_name))
    
    # c100 sln-mo-lc test acc
    dataset_name, model, noise_mode, if_train = "cifar100", "sln-mo-lc", "symmetric", False
    c100_sln_mo_lc_sym_acc, c100_sln_mo_lc_sym_acc_mean, c100_sln_mo_lc_sym_acc_std = get_accuracy(dataset_name, model, noise_mode, if_train)

    dataset_name, model, noise_mode, if_train = "cifar100", "sln-mo-lc", "asymmetric", False
    c100_sln_mo_lc_asym_acc, c100_sln_mo_lc_asym_acc_mean, c100_sln_mo_lc_asym_acc_std = get_accuracy(dataset_name, model, noise_mode, if_train)

    dataset_name, model, noise_mode, if_train = "cifar100", "sln-mo-lc", "dependent", False
    c100_sln_mo_lc_dep_acc, c100_sln_mo_lc_dep_acc_mean, c100_sln_mo_lc_dep_acc_std = get_accuracy(dataset_name, model, noise_mode, if_train)

    plt.figure(figsize=(6, 4))
    plt.plot(c100_sln_mo_lc_sym_acc*100, color="blue", label="sym")
    plt.plot(c100_sln_mo_lc_asym_acc*100, color="orange", label="asym")
    plt.plot(c100_sln_mo_lc_dep_acc*100, color="green", label="dep")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Test Accuracy (SLN-MO-LC) on CIFAR-100")
    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    plt.xlim(0, 300)
    plt.ylim(10, 72)
    plt.tight_layout()
    fig_name = "c100-sln-mo-lc-test-acc.png"
    plt.savefig(os.path.join(results_path, fig_name))
    
    plt.figure(figsize=(6, 4))
    plt.plot(c100_sln_mo_lc_sym_acc*100, color="blue", label="sym")
    plt.plot(c100_sln_mo_lc_asym_acc*100, color="orange", label="asym")
    plt.plot(c100_sln_mo_lc_dep_acc*100, color="green", label="dep")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Test Accuracy (SLN-MO-LC) on CIFAR-100")
    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    plt.xlim(220, 300)
    plt.ylim(40, 72)
    plt.tight_layout()
    fig_name = "c100-sln-mo-lc-test-acc-zoom.png"
    plt.savefig(os.path.join(results_path, fig_name))
    
    # c100 test acc table
    data_cifar100 = {
        "model": ["ce", "sln", "sln-mo", "sln-mo-lc"],
        "symmetric": [
            str(round(c100_ce_sym_acc_mean*100,2))+u"\u00B1"+str(round(c100_ce_sym_acc_std*100,2)),
            str(round(c100_sln_sym_acc_mean*100,2))+u"\u00B1"+str(round(c100_sln_sym_acc_std*100,2)),
            str(round(c100_sln_mo_sym_acc_mean*100,2))+u"\u00B1"+str(round(c100_sln_mo_sym_acc_std*100,2)),
            str(round(c100_sln_mo_lc_sym_acc_mean*100,2))+u"\u00B1"+str(round(c100_sln_mo_lc_sym_acc_std*100,2)),
        ],
        "asymmetric": [
            str(round(c100_ce_asym_acc_mean*100,2))+u"\u00B1"+str(round(c100_ce_asym_acc_std*100,2)),
            str(round(c100_sln_asym_acc_mean*100,2))+u"\u00B1"+str(round(c100_sln_asym_acc_std*100,2)),
            str(round(c100_sln_mo_asym_acc_mean*100,2))+u"\u00B1"+str(round(c100_sln_mo_asym_acc_std*100,2)),
            str(round(c100_sln_mo_lc_asym_acc_mean*100,2))+u"\u00B1"+str(round(c100_sln_mo_lc_asym_acc_std*100,2)),
        ],
        "dependent": [
            str(round(c100_ce_dep_acc_mean*100,2))+u"\u00B1"+str(round(c100_ce_dep_acc_std*100,2)),
            str(round(c100_sln_dep_acc_mean*100,2))+u"\u00B1"+str(round(c100_sln_dep_acc_std*100,2)),
            str(round(c100_sln_mo_dep_acc_mean*100,2))+u"\u00B1"+str(round(c100_sln_mo_dep_acc_std*100,2)),
            str(round(c100_sln_mo_lc_dep_acc_mean*100,2))+u"\u00B1"+str(round(c100_sln_mo_lc_dep_acc_std*100,2)),
        ],
    }
    df_table_cifar100 = pd.DataFrame(data=data_cifar100)
    table_name = "c100-test-acc-table.csv"
    df_table_cifar100.to_csv(os.path.join(results_path, table_name))
    
    # c100 ce train losses
    dataset_name, model, noise_mode, loss_type = "cifar100", "ce", "symmetric", "clean"
    c100_ce_sym_loss_train_clean = get_loss_train(dataset_name, model, noise_mode, loss_type)
    dataset_name, model, noise_mode, loss_type = "cifar100", "ce", "symmetric", "noisy"
    c100_ce_sym_loss_train_noisy = get_loss_train(dataset_name, model, noise_mode, loss_type)

    dataset_name, model, noise_mode, loss_type = "cifar100", "ce", "asymmetric", "clean"
    c100_ce_asym_loss_train_clean = get_loss_train(dataset_name, model, noise_mode, loss_type)
    dataset_name, model, noise_mode, loss_type = "cifar100", "ce", "asymmetric", "noisy"
    c100_ce_asym_loss_train_noisy = get_loss_train(dataset_name, model, noise_mode, loss_type)

    dataset_name, model, noise_mode, loss_type = "cifar100", "ce", "dependent", "clean"
    c100_ce_dep_loss_train_clean = get_loss_train(dataset_name, model, noise_mode, loss_type)
    dataset_name, model, noise_mode, loss_type = "cifar100", "ce", "dependent", "noisy"
    c100_ce_dep_loss_train_noisy = get_loss_train(dataset_name, model, noise_mode, loss_type)


    plt.figure(figsize=(6, 4))
    plt.plot(c100_ce_sym_loss_train_clean, color="blue", label="sym clean")
    plt.plot(c100_ce_sym_loss_train_noisy, color="blue", linestyle='dashed', label="sym noisy")
    plt.plot(c100_ce_asym_loss_train_clean, color="orange", label="asym clean")
    plt.plot(c100_ce_asym_loss_train_noisy, color="orange", linestyle='dashed', label="asym noisy")
    plt.plot(c100_ce_dep_loss_train_clean, color="green", label="dep clean")
    plt.plot(c100_ce_dep_loss_train_noisy, color="green", linestyle='dashed', label="dep noisy")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CE Training Loss (CE) on CIFAR-100")
    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    plt.xlim(0, 300)
    plt.ylim(0, 5.5)
    plt.tight_layout()
    fig_name = "c100-ce-train-loss.png"
    plt.savefig(os.path.join(results_path, fig_name))
    
    # c100 sln train losses

    dataset_name, model, noise_mode, loss_type = "cifar100", "sln", "symmetric", "clean"
    c100_sln_sym_loss_train_clean = get_loss_train(dataset_name, model, noise_mode, loss_type)
    dataset_name, model, noise_mode, loss_type = "cifar100", "sln", "symmetric", "noisy"
    c100_sln_sym_loss_train_noisy = get_loss_train(dataset_name, model, noise_mode, loss_type)

    dataset_name, model, noise_mode, loss_type = "cifar100", "sln", "asymmetric", "clean"
    c100_sln_asym_loss_train_clean = get_loss_train(dataset_name, model, noise_mode, loss_type)
    dataset_name, model, noise_mode, loss_type = "cifar100", "sln", "asymmetric", "noisy"
    c100_sln_asym_loss_train_noisy = get_loss_train(dataset_name, model, noise_mode, loss_type)

    dataset_name, model, noise_mode, loss_type = "cifar100", "sln", "dependent", "clean"
    c100_sln_dep_loss_train_clean = get_loss_train(dataset_name, model, noise_mode, loss_type)
    dataset_name, model, noise_mode, loss_type = "cifar100", "sln", "dependent", "noisy"
    c100_sln_dep_loss_train_noisy = get_loss_train(dataset_name, model, noise_mode, loss_type)

    plt.figure(figsize=(6, 4))
    plt.plot(c100_sln_sym_loss_train_clean, color="blue", label="sym clean")
    plt.plot(c100_sln_sym_loss_train_noisy, color="blue", linestyle='dashed', label="sym noisy")
    plt.plot(c100_sln_asym_loss_train_clean, color="orange", label="asym clean")
    plt.plot(c100_sln_asym_loss_train_noisy, color="orange", linestyle='dashed', label="asym noisy")
    plt.plot(c100_sln_dep_loss_train_clean, color="green", label="dep clean")
    plt.plot(c100_sln_dep_loss_train_noisy, color="green", linestyle='dashed', label="dep noisy")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CE Training Loss (SLN) on CIFAR-100")
    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    plt.xlim(0, 300)
    plt.ylim(0, 5.5)
    plt.tight_layout()
    fig_name = "c100-sln-train-loss.png"
    plt.savefig(os.path.join(results_path, fig_name))
    
    # c100 sln-mo train losses

    dataset_name, model, noise_mode, loss_type = "cifar100", "sln-mo", "symmetric", "clean"
    c100_sln_mo_sym_loss_train_clean = get_loss_train(dataset_name, model, noise_mode, loss_type)
    dataset_name, model, noise_mode, loss_type = "cifar100", "sln-mo", "symmetric", "noisy"
    c100_sln_mo_sym_loss_train_noisy = get_loss_train(dataset_name, model, noise_mode, loss_type)

    dataset_name, model, noise_mode, loss_type = "cifar100", "sln-mo", "asymmetric", "clean"
    c100_sln_mo_asym_loss_train_clean = get_loss_train(dataset_name, model, noise_mode, loss_type)
    dataset_name, model, noise_mode, loss_type = "cifar100", "sln-mo", "asymmetric", "noisy"
    c100_sln_mo_asym_loss_train_noisy = get_loss_train(dataset_name, model, noise_mode, loss_type)

    dataset_name, model, noise_mode, loss_type = "cifar100", "sln-mo", "dependent", "clean"
    c100_sln_mo_dep_loss_train_clean = get_loss_train(dataset_name, model, noise_mode, loss_type)
    dataset_name, model, noise_mode, loss_type = "cifar100", "sln-mo", "dependent", "noisy"
    c100_sln_mo_dep_loss_train_noisy = get_loss_train(dataset_name, model, noise_mode, loss_type)

    plt.figure(figsize=(6, 4))
    plt.plot(c100_sln_mo_sym_loss_train_clean, color="blue", label="sym clean")
    plt.plot(c100_sln_mo_sym_loss_train_noisy, color="blue", linestyle='dashed', label="sym noisy")
    plt.plot(c100_sln_mo_asym_loss_train_clean, color="orange", label="asym clean")
    plt.plot(c100_sln_mo_asym_loss_train_noisy, color="orange", linestyle='dashed', label="asym noisy")
    plt.plot(c100_sln_mo_dep_loss_train_clean, color="green", label="dep clean")
    plt.plot(c100_sln_mo_dep_loss_train_noisy, color="green", linestyle='dashed', label="dep noisy")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CE Training Loss (SLN-MO) on CIFAR-100")
    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    plt.xlim(0, 300)
    plt.ylim(0, 5.5)
    plt.tight_layout()
    fig_name = "c100-sln-mo-train-loss.png"
    plt.savefig(os.path.join(results_path, fig_name))
    
    # c100 sln-mo-lc train losses

    dataset_name, model, noise_mode, loss_type = "cifar100", "sln-mo-lc", "symmetric", "clean"
    c100_sln_mo_lc_sym_loss_train_clean = get_loss_train(dataset_name, model, noise_mode, loss_type)
    dataset_name, model, noise_mode, loss_type = "cifar100", "sln-mo-lc", "symmetric", "noisy"
    c100_sln_mo_lc_sym_loss_train_noisy = get_loss_train(dataset_name, model, noise_mode, loss_type)

    dataset_name, model, noise_mode, loss_type = "cifar100", "sln-mo-lc", "asymmetric", "clean"
    c100_sln_mo_lc_asym_loss_train_clean = get_loss_train(dataset_name, model, noise_mode, loss_type)
    dataset_name, model, noise_mode, loss_type = "cifar100", "sln-mo-lc", "asymmetric", "noisy"
    c100_sln_mo_lc_asym_loss_train_noisy = get_loss_train(dataset_name, model, noise_mode, loss_type)

    dataset_name, model, noise_mode, loss_type = "cifar100", "sln-mo-lc", "dependent", "clean"
    c100_sln_mo_lc_dep_loss_train_clean = get_loss_train(dataset_name, model, noise_mode, loss_type)
    dataset_name, model, noise_mode, loss_type = "cifar100", "sln-mo-lc", "dependent", "noisy"
    c100_sln_mo_lc_dep_loss_train_noisy = get_loss_train(dataset_name, model, noise_mode, loss_type)

    plt.figure(figsize=(6, 4))
    plt.plot(c100_sln_mo_lc_sym_loss_train_clean, color="blue", label="sym clean")
    plt.plot(c100_sln_mo_lc_sym_loss_train_noisy, color="blue", linestyle='dashed', label="sym noisy")
    plt.plot(c100_sln_mo_lc_asym_loss_train_clean, color="orange", label="asym clean")
    plt.plot(c100_sln_mo_lc_asym_loss_train_noisy, color="orange", linestyle='dashed', label="asym noisy")
    plt.plot(c100_sln_mo_lc_dep_loss_train_clean, color="green", label="dep clean")
    plt.plot(c100_sln_mo_lc_dep_loss_train_noisy, color="green", linestyle='dashed', label="dep noisy")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CE Training Loss (SLN-MO-LC) on CIFAR-100")
    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    plt.xlim(0, 300)
    plt.ylim(0, 5.5)
    plt.tight_layout()
    fig_name = "c100-sln-mo-lc-train-loss.png"
    plt.savefig(os.path.join(results_path, fig_name))
    
    # Animal-10N
    
    animal_ce_acc, animal_ce_mean, animal_ce_std = get_accuracy_animal(model="ce", if_train=False)
    animal_sln_acc, animal_sln_mean, animal_sln_std = get_accuracy_animal(model="sln", if_train=False)
    animal_sln_mo_acc, animal_sln_mo_mean, animal_sln_mo_std = get_accuracy_animal(model="sln-mo", if_train=False)
    animal_sln_mo_lc_acc, animal_sln_mo_lc_mean, animal_sln_mo_lc_std = get_accuracy_animal(model="sln-mo-lc", if_train=False)

    plt.figure(figsize=(6, 4))
    plt.plot(animal_ce_acc*100, color="red", label="ce")
    plt.plot(animal_sln_acc*100, color="blue", label="sln")
    plt.plot(animal_sln_mo_acc*100, color="orange", label="sln-mo")
    plt.plot(animal_sln_mo_lc_acc*100, color="green", label="sln-mo-lc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy on Animal-10n")
    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    plt.xlim(0, 300)
    plt.ylim(50, 80)
    plt.tight_layout()
    fig_name = "animal10n-test-acc.png"
    plt.savefig(os.path.join(results_path, fig_name))
    
    plt.figure(figsize=(6, 4))
    plt.plot(animal_ce_acc*100, color="red", label="ce")
    plt.plot(animal_sln_acc*100, color="blue", label="sln")
    plt.plot(animal_sln_mo_acc*100, color="orange", label="sln-mo")
    plt.plot(animal_sln_mo_lc_acc*100, color="green", label="sln-mo-lc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy on Animal-10n")
    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    plt.xlim(220, 300)
    plt.ylim(70, 80)
    plt.tight_layout()
    fig_name = "animal10n-test-acc-zoom.png"
    plt.savefig(os.path.join(results_path, fig_name))
    
    # animal test acc table
    data_animal = {
        "model": ["ce", "sln", "sln-mo", "sln-mo-lc"],
        "real noise": [
            str(round(animal_ce_mean*100,2))+u"\u00B1"+str(round(animal_ce_std*100,2)),
            str(round(animal_sln_mean*100,2))+u"\u00B1"+str(round(animal_sln_std*100,2)),
            str(round(animal_sln_mo_mean*100,2))+u"\u00B1"+str(round(animal_sln_mo_std*100,2)),
            str(round(animal_sln_mo_lc_mean*100,2))+u"\u00B1"+str(round(animal_sln_mo_lc_std*100,2)),
        ],
    }
    df_table_animal = pd.DataFrame(data=data_animal)
    table_name = "animal10n-test-acc-table.csv"
    df_table_animal.to_csv(os.path.join(results_path, table_name))
