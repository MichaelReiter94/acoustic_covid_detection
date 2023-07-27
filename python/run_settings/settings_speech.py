import os
# MIL settings
# parameters = dict(
#     batch=[10],
#     lr=[1e-5],
#     wd=[1e-4],
#     lr_decay=[0.85],
#     mixup_a=[0],
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
#     dropout_p=[0.15],
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
# LOAD_FROM_DISC = r"data\Coswara_processed\models\2023-07-06_epoch83_evalMetric_85.7_seed99468865_speech.pth"
LOAD_FROM_DISC_multipleSplits = None
# LOAD_FROM_DISC_multipleSplits = [
#     r"data/Coswara_processed/models/2023-07-11_epoch124_evalMetric_85.4_combined_speech _seed99468865.pth",
#     r"data/Coswara_processed/models/2023-07-11_epoch137_evalMetric_82.6_combined_speech _seed215674.pth",
#     r"data/Coswara_processed/models/2023-07-11_epoch136_evalMetric_83.8_combined_speech _seed3213213211.pth",
#     r"data/Coswara_processed/models/2023-07-11_epoch113_evalMetric_85.0_combined_speech _seed55555555.pth",
#     r"data/Coswara_processed/models/2023-07-11_epoch21_evalMetric_85.4_combined_speech _seed66445511337.pth"]
# ###########################################   DO NOT CHANGE LINES BELOW   ############################################
DATASET_NAME = "2023_06_25_logmel_combined_speech_NEW_23msHop_46msFFT_fmax11000_224logmel"
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