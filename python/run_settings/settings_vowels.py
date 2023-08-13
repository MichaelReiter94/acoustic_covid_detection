import os
# MIL settings
# parameters = dict(
#     batch=[20],
#     lr=[1e-5],
#     wd=[1e-4],
#     lr_decay=[0.75],
#     mixup_a=[0.0],
#     mixup_p=[0],
#     use_augm_datasets=[False],
#     shift=[False],
#     sigma=[0],
#     weighted_sampler=[False],
#     class_weight=[1],
#     bag_size=[12],
#     n_MIL_Neurons=[64],
#     time_steps=[336],
#     lr_in=[None],
#     lr_mil=[1],
#     dropout_p=[0.1],
#     focal_loss=[0],
#     exclude_outliers=[0],
#     use_resnorm=[False],
#     val_oversampl=[8],
#     exclude_conf_miscl=[True],
#     transfer_func_sim=[0],
#     random_gain=[0]
# )

parameters = dict(
    batch=[64],
    lr=[1e-4, 8e-5],
    lr_decay=[0.97],
    lr_in=[None],
    wd=[1e-4],

    normalize=[False],
    time_steps=[336],
    use_augm_datasets=[False],

    shift=[True],
    sigma=[0],
    mixup_a=[0.05, 0.2, 0.4],
    mixup_p=[0, 1],
    dropout_p=[0],
    transfer_func_sim=[6],
    random_gain=[3],

    exclude_outliers=[0],
    use_resnorm=[False],
    focal_loss=[0],

    bag_size=[None],
    lr_mil=[None],
    n_MIL_Neurons=[None],

    exclude_conf_miscl=[False],
    self_assessment_penalty=[0.6],

    val_oversampl=[8],
    class_weight=[1],
    weighted_sampler=[True],

    time_domain_augmentations_pos=[0],
    time_domain_augmentations_neg=[0],
    exclude_exposed=[True]  # no effect, just a reminder to include into the excel sheet
)

USE_MIL = False

RUN_COMMENT = f"mixup"
n_epochs = 130
n_cross_validation_runs = 3

SAVE_TO_DISC = False
EVALUATE_TEST_SET = False
ID_PERFORMANCE_TRACKING = None
LOAD_FROM_DISC = False
# LOAD_FROM_DISC = r"data\Coswara_processed\models\2023-07-06_epoch91_evalMetric_83.7_seed99468865_vowels.pth"
LOAD_FROM_DISC_multipleSplits = None
FREEZE_MODEL = False
# ###########################################   DO NOT CHANGE LINES BELOW   ############################################
DATASET_NAME = "2023_06_25_logmel_combined_vowels_NEW_23msHop_96msFFT_fmax11000_224logmel"
MODEL_NAME = "resnet18"
QUICK_TRAIN_FOR_TESTS = False
SAMPLES_PER_EPOCH = 1024
TRAIN_ON_FULL_SET = False
USE_TRAIN_VAL_TEST_SPLIT = True
