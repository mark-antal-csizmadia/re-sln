import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


parts = {
    "loss_train_all": "_loss_train_all-tag-loss_train",
    "loss_train_clean": "_loss_train_clean-tag-loss_train",
    "loss_train_noisy": "_loss_train_noisy-tag-loss_train",
    "loss_test": "-tag-loss_test",
    "accuracy_train": "-tag-accuracy_train",
    "accuracy_test": "-tag-accuracy_test"
}


def get_list(tb_logs_path, exp_id, part):
    return np.array(pd.read_csv(os.path.join(tb_logs_path, get_csv_name(exp_id=exp_id, part=part)))["Value"])


def get_csv_name(exp_id, part):
    return "run-" + exp_id + part + ".csv"


def fig_1_custom(results_path):
    """Sym, asym and openset are custom, dependent from paper"""
    tb_logs_path = os.path.join(results_path, "tb_logs")

    models = {
        "ce":
            {
                "symmetric": ["exp_2021-11-25 13_17_26.851200","exp_2021-11-25 20_30_28.794160"],
                "asymmetric": ["exp_2021-11-26 09_21_19.524188","exp_2021-11-26 14_03_18.975684"],
                "dependent": ["exp_2021-11-26 20_14_40.983299"],
                "openset": ["exp_2021-11-27 13_35_23.026659"]
            },
        "sln":
            {
                "symmetric": ["exp_2021-11-25 15_38_09.361059","exp_2021-11-25 20_31_37.546765"],
                "asymmetric": ["exp_2021-11-26 11_41_56.758060","exp_2021-11-26 16_11_00.844488"],
                "dependent": ["exp_2021-11-27 11_07_55.847340"],
                "openset": ["exp_2021-11-27 13_44_37.885816"]
            }
    }

    # Fig 1 CE Loss (CE), custom+paper
    ce_sym_loss_clean = \
        (get_list(tb_logs_path=tb_logs_path, exp_id=models["ce"]["symmetric"][0], part=parts["loss_train_clean"]) +
        get_list(tb_logs_path=tb_logs_path, exp_id=models["ce"]["symmetric"][1], part=parts["loss_train_clean"]))/2
    ce_sym_loss_noisy = \
        (get_list(tb_logs_path=tb_logs_path, exp_id=models["ce"]["symmetric"][0], part=parts["loss_train_noisy"]) +
         get_list(tb_logs_path=tb_logs_path, exp_id=models["ce"]["symmetric"][1], part=parts["loss_train_noisy"]))/2
    ce_asym_loss_clean = \
        (get_list(tb_logs_path=tb_logs_path, exp_id=models["ce"]["asymmetric"][0], part=parts["loss_train_clean"]) +
         get_list(tb_logs_path=tb_logs_path, exp_id=models["ce"]["asymmetric"][1], part=parts["loss_train_clean"]))/2
    ce_asym_loss_noisy = \
        (get_list(tb_logs_path=tb_logs_path, exp_id=models["ce"]["asymmetric"][0], part=parts["loss_train_noisy"]) +
         get_list(tb_logs_path=tb_logs_path, exp_id=models["ce"]["asymmetric"][1], part=parts["loss_train_noisy"]))/2
    ce_dependent_loss_clean = \
        get_list(tb_logs_path=tb_logs_path, exp_id=models["ce"]["dependent"][0], part=parts["loss_train_clean"])
    ce_dependent_loss_noisy = \
        get_list(tb_logs_path=tb_logs_path, exp_id=models["ce"]["dependent"][0], part=parts["loss_train_noisy"])
    ce_openset_loss_clean = \
        get_list(tb_logs_path=tb_logs_path, exp_id=models["ce"]["openset"][0], part=parts["loss_train_clean"])
    ce_openset_loss_noisy = \
        get_list(tb_logs_path=tb_logs_path, exp_id=models["ce"]["openset"][0], part=parts["loss_train_noisy"])

    plt.figure(figsize=(6, 4))
    plt.plot(ce_sym_loss_clean, color="b", label="sym clean")
    plt.plot(ce_sym_loss_noisy, color="b", linestyle='dashed', label="sym noisy")
    plt.plot(ce_asym_loss_clean, color="orange", label="asym clean")
    plt.plot(ce_asym_loss_noisy, color="orange", linestyle='dashed', label="asym noisy")
    plt.plot(ce_dependent_loss_clean, color="green", label="dep clean")
    plt.plot(ce_dependent_loss_noisy, color="green", linestyle='dashed', label="dep noisy")
    plt.plot(ce_openset_loss_clean, color="red", label="openset clean")
    plt.plot(ce_openset_loss_noisy, color="red", linestyle='dashed', label="openset noisy")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CE Training Loss (CE), custom+paper noise")
    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    plt.xlim(0, 300)
    plt.ylim(0, 3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, "ce_loss_ce_c_p.png"))
    #plt.show()

    # Fig 1 CE Loss (SLN), custom+paper
    sln_sym_loss_clean = \
        (get_list(tb_logs_path=tb_logs_path, exp_id=models["sln"]["symmetric"][0], part=parts["loss_train_clean"]) +
         get_list(tb_logs_path=tb_logs_path, exp_id=models["sln"]["symmetric"][1], part=parts["loss_train_clean"]))/2
    sln_sym_loss_noisy = \
        (get_list(tb_logs_path=tb_logs_path, exp_id=models["sln"]["symmetric"][0], part=parts["loss_train_noisy"]) +
         get_list(tb_logs_path=tb_logs_path, exp_id=models["sln"]["symmetric"][1], part=parts["loss_train_noisy"]))/2
    sln_asym_loss_clean = \
        (get_list(tb_logs_path=tb_logs_path, exp_id=models["sln"]["asymmetric"][0], part=parts["loss_train_clean"]) +
         get_list(tb_logs_path=tb_logs_path, exp_id=models["sln"]["asymmetric"][1], part=parts["loss_train_clean"]))/2
    sln_asym_loss_noisy = \
        (get_list(tb_logs_path=tb_logs_path, exp_id=models["sln"]["asymmetric"][0], part=parts["loss_train_noisy"]) +
         get_list(tb_logs_path=tb_logs_path, exp_id=models["sln"]["asymmetric"][1], part=parts["loss_train_noisy"]))/2
    sln_dependent_loss_clean = \
        get_list(tb_logs_path=tb_logs_path, exp_id=models["sln"]["dependent"][0], part=parts["loss_train_clean"])
    sln_dependent_loss_noisy = \
        get_list(tb_logs_path=tb_logs_path, exp_id=models["sln"]["dependent"][0], part=parts["loss_train_noisy"])
    sln_openset_loss_clean = \
        get_list(tb_logs_path=tb_logs_path, exp_id=models["sln"]["openset"][0], part=parts["loss_train_clean"])
    sln_openset_loss_noisy = \
        get_list(tb_logs_path=tb_logs_path, exp_id=models["sln"]["openset"][0], part=parts["loss_train_noisy"])

    plt.figure(figsize=(6, 4))
    plt.plot(sln_sym_loss_clean, color="b", label="sym clean")
    plt.plot(sln_sym_loss_noisy, color="b", linestyle='dashed', label="sym noisy")
    plt.plot(sln_asym_loss_clean, color="orange", label="asym clean")
    plt.plot(sln_asym_loss_noisy, color="orange", linestyle='dashed', label="asym noisy")
    plt.plot(sln_dependent_loss_clean, color="green", label="dep clean")
    plt.plot(sln_dependent_loss_noisy, color="green", linestyle='dashed', label="dep noisy")
    plt.plot(sln_openset_loss_clean, color="red", label="openset clean")
    plt.plot(sln_openset_loss_noisy, color="red", linestyle='dashed', label="openset noisy")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CE Training Loss (SLN), custom+paper noise")
    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    plt.xlim(0, 300)
    plt.ylim(0, 3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, "ce_loss_sln_c_p.png"))
    # plt.show()

    # Fig 1 Test Accuracy (CE), custom+paper
    ce_sym_acc_clean = \
        (get_list(tb_logs_path=tb_logs_path, exp_id=models["ce"]["symmetric"][0], part=parts["accuracy_test"]) +
         get_list(tb_logs_path=tb_logs_path, exp_id=models["ce"]["symmetric"][1], part=parts["accuracy_test"]))/2*100
    ce_sym_acc_noisy = \
        (get_list(tb_logs_path=tb_logs_path, exp_id=models["ce"]["symmetric"][0], part=parts["accuracy_test"]) +
         get_list(tb_logs_path=tb_logs_path, exp_id=models["ce"]["symmetric"][1], part=parts["accuracy_test"]))/2*100
    ce_asym_acc_clean = \
        (get_list(tb_logs_path=tb_logs_path, exp_id=models["ce"]["asymmetric"][0], part=parts["accuracy_test"]) +
         get_list(tb_logs_path=tb_logs_path, exp_id=models["ce"]["asymmetric"][1], part=parts["accuracy_test"]))/2*100
    ce_asym_acc_noisy = \
        (get_list(tb_logs_path=tb_logs_path, exp_id=models["ce"]["asymmetric"][0], part=parts["accuracy_test"]) +
         get_list(tb_logs_path=tb_logs_path, exp_id=models["ce"]["asymmetric"][1], part=parts["accuracy_test"]))/2*100
    ce_dependent_acc_clean = \
        get_list(tb_logs_path=tb_logs_path, exp_id=models["ce"]["dependent"][0], part=parts["accuracy_test"])*100
    ce_dependent_acc_noisy = \
        get_list(tb_logs_path=tb_logs_path, exp_id=models["ce"]["dependent"][0], part=parts["accuracy_test"])*100
    ce_openset_acc_clean = \
        get_list(tb_logs_path=tb_logs_path, exp_id=models["ce"]["openset"][0], part=parts["accuracy_test"])*100
    ce_openset_acc_noisy = \
        get_list(tb_logs_path=tb_logs_path, exp_id=models["ce"]["openset"][0], part=parts["accuracy_test"])*100

    plt.figure(figsize=(6, 4))
    plt.plot(list(ce_sym_acc_clean), color="b", label="sym clean")
    plt.plot(ce_sym_acc_noisy, color="b", linestyle='dashed', label="sym noisy")
    plt.plot(ce_asym_acc_clean, color="orange", label="asym clean")
    plt.plot(ce_asym_acc_noisy, color="orange", linestyle='dashed', label="asym noisy")
    plt.plot(ce_dependent_acc_clean, color="green", label="dep clean")
    plt.plot(ce_dependent_acc_noisy, color="green", linestyle='dashed', label="dep noisy")
    plt.plot(ce_openset_acc_clean, color="red", label="openset clean")
    plt.plot(ce_openset_acc_noisy, color="red", linestyle='dashed', label="openset noisy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Test Accuracy (CE), custom+paper noise")
    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    plt.xlim(0, 300)
    plt.ylim(50, 87)
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, "acc_ce_c_p.png"))
    #plt.show()

    # Fig 1 Test Accuracy (SLN), custom+paper
    sln_sym_acc_clean = \
        (get_list(tb_logs_path=tb_logs_path, exp_id=models["sln"]["symmetric"][0], part=parts["accuracy_test"]) +
         get_list(tb_logs_path=tb_logs_path, exp_id=models["sln"]["symmetric"][1], part=parts["accuracy_test"]))/2*100
    sln_sym_acc_noisy = \
        (get_list(tb_logs_path=tb_logs_path, exp_id=models["sln"]["symmetric"][0], part=parts["accuracy_test"]) +
         get_list(tb_logs_path=tb_logs_path, exp_id=models["sln"]["symmetric"][1], part=parts["accuracy_test"]))/2*100
    sln_asym_acc_clean = \
        (get_list(tb_logs_path=tb_logs_path, exp_id=models["sln"]["asymmetric"][0], part=parts["accuracy_test"]) +
         get_list(tb_logs_path=tb_logs_path, exp_id=models["sln"]["asymmetric"][1], part=parts["accuracy_test"]))/2*100
    sln_asym_acc_noisy = \
        (get_list(tb_logs_path=tb_logs_path, exp_id=models["sln"]["asymmetric"][0], part=parts["accuracy_test"]) +
         get_list(tb_logs_path=tb_logs_path, exp_id=models["sln"]["asymmetric"][1], part=parts["accuracy_test"]))/2*100
    sln_dependent_acc_clean = \
        get_list(tb_logs_path=tb_logs_path, exp_id=models["sln"]["dependent"][0], part=parts["accuracy_test"])*100
    sln_dependent_acc_noisy = \
        get_list(tb_logs_path=tb_logs_path, exp_id=models["sln"]["dependent"][0], part=parts["accuracy_test"])*100
    sln_openset_acc_clean = \
        get_list(tb_logs_path=tb_logs_path, exp_id=models["sln"]["openset"][0], part=parts["accuracy_test"])*100
    sln_openset_acc_noisy = \
        get_list(tb_logs_path=tb_logs_path, exp_id=models["sln"]["openset"][0], part=parts["accuracy_test"])*100

    plt.figure(figsize=(6, 4))
    plt.plot(list(sln_sym_acc_clean), color="b", label="sym clean")
    plt.plot(sln_sym_acc_noisy, color="b", linestyle='dashed', label="sym noisy")
    plt.plot(sln_asym_acc_clean, color="orange", label="asym clean")
    plt.plot(sln_asym_acc_noisy, color="orange", linestyle='dashed', label="asym noisy")
    plt.plot(sln_dependent_acc_clean, color="green", label="dep clean")
    plt.plot(sln_dependent_acc_noisy, color="green", linestyle='dashed', label="dep noisy")
    plt.plot(sln_openset_acc_clean, color="red", label="openset clean")
    plt.plot(sln_openset_acc_noisy, color="red", linestyle='dashed', label="openset noisy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Test Accuracy (SLN), custom+paper noise")
    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    plt.xlim(0, 300)
    plt.ylim(50, 87)
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, "acc_sln_c_p.png"))
    # plt.show()


if __name__ == "__main__":
    results_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "results")
    fig_1_custom_results_path = os.path.join(results_path, "fig_1_custom")
    fig_1_custom(results_path=fig_1_custom_results_path)
