import os
# MIL settings
# parameters = dict(
#     batch=[10],
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
#     time_steps=[224],
#     lr_in=[None],
#     lr_mil=[1],
#     dropout_p=[0.35],
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
    lr=[1e-3, 3e-4, 1e-4, 3e-5, 1e-5],
    lr_decay=[0.97],
    lr_in=[None],
    wd=[1e-4],

    normalize=[True, False],
    time_steps=[336],
    use_augm_datasets=[False],

    shift=[False],
    sigma=[0],
    mixup_a=[0],
    mixup_p=[0],
    dropout_p=[0],
    transfer_func_sim=[0],
    random_gain=[0],

    exclude_outliers=[0],
    use_resnorm=[False],
    focal_loss=[0],

    bag_size=[None],
    lr_mil=[None],
    n_MIL_Neurons=[None],

    exclude_conf_miscl=[False],

    val_oversampl=[8],
    class_weight=[1],
    weighted_sampler=[True],
)

USE_MIL = False

RUN_COMMENT = f"baseline_pretrained_normalizedInput"
n_epochs = 75
n_cross_validation_runs = 5

SAVE_TO_DISC = False
EVALUATE_TEST_SET = False
ID_PERFORMANCE_TRACKING = None

LOAD_FROM_DISC = False
# LOAD_FROM_DISC = r"data\Coswara_processed\models\2023-06-12_resnet18_23ms_82_7_AUCROC_iter74.pth"
# LOAD_FROM_DISC = r"data\Coswara_processed\models\2023-07-11_epoch102_evalMetric_81.9_combined_breaths _seed99468865.pth"
LOAD_FROM_DISC_multipleSplits = None

# ###########################################   DO NOT CHANGE LINES BELOW   ############################################
DATASET_NAME = "logmel_combined_breaths_NEW_46msHop_92msFFT_fmax11000_224logmel"
MODEL_NAME = "resnet18"
QUICK_TRAIN_FOR_TESTS = False
SAMPLES_PER_EPOCH = 1024
TRAIN_ON_FULL_SET = False
USE_TRAIN_VAL_TEST_SPLIT = True

if isinstance(LOAD_FROM_DISC, str):
    LOAD_FROM_DISC = os.path.join(*LOAD_FROM_DISC.split("\\"))
if TRAIN_ON_FULL_SET:
    RUN_COMMENT += "_trainOnFullSet"
    EVALUATE_TEST_SET = False
