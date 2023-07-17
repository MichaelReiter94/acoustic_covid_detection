import os

# MIL settings
parameters = dict(
    batch=[10],
    lr=[1e-5],
    wd=[1e-4],
    lr_decay=[0.2],
    mixup_a=[0.2],
    mixup_p=[0.8],
    use_augm_datasets=[False],
    shift=[True],
    sigma=[0.2],
    weighted_sampler=[True],
    class_weight=[1],
    bag_size=[12],
    n_MIL_Neurons=[64],
    time_steps=[336],
    lr_in=[None],
    lr_mil=[1e0],
    dropout_p=[0.1],
    focal_loss=[0],
    exclude_outliers=[0],
    use_resnorm=[False],
    val_oversampl=[8]
)

# evaluation settings for saved models
# parameters = dict(
#     batch=[64],
#     lr=[7e-10],
#     wd=[1e-4],
#     lr_decay=[0.99],
#     mixup_a=[0.4],
#     mixup_p=[0.8],
#     use_augm_datasets=[False],
#     shift=[True],
#     sigma=[0.2],
#     weighted_sampler=[True],
#     class_weight=[1],
#     bag_size=[12],
#     n_MIL_Neurons=[32],
#     time_steps=[336],
#     lr_in=[None],
#     lr_mil=[1e3],
#     dropout_p=[0],
#     focal_loss=[0],
#     exclude_outliers=[0],
#     use_resnorm=[False],
#     val_oversampl=[1]
# )
# if USE_MIL is True, VAL_SET_OVERSAMPLING_FACTOR will be set to be 1 (1 means no oversampling)
# VAL_SET_OVERSAMPLING_FACTOR = parameters["val_oversampl"]

# DATASET_NAME = "2023_06_25_logmel_combined_vowels_NEW_23msHop_96msFFT_fmax11000_224logmel"
# DATASET_NAME = "2023_06_25_logmel_combined_speech_NEW_23msHop_46msFFT_fmax11000_224logmel"
DATASET_NAME = "logmel_combined_breaths_NEW_23msHop_46msFFT_fmax11000_224logmel"
# DATASET_NAME = "2023_05_22_logmel_combined_coughs_NEW_11msHop_23msFFT_fmax11000_224logmel"


RUN_COMMENT = f""
n_epochs = 3
n_cross_validation_runs = 1
USE_MIL = True
MODEL_NAME = "resnet18"

USE_TRAIN_VAL_TEST_SPLIT = True
QUICK_TRAIN_FOR_TESTS = False
SAMPLES_PER_EPOCH = 256

# LOAD_FROM_DISC = False
# LOAD_FROM_DISC = r"data\Coswara_processed\models\2023-07-02_epoch104_evalMetric_77.9_seed99468865.pth"
# LOAD_FROM_DISC = r"data\Coswara_processed\models\2023-07-02_epoch119_evalMetric_81.9_seed215674.pth"

# LOAD_FROM_DISC = r"data\Coswara_processed\models\2023-07-06_epoch83_evalMetric_85.7_seed99468865_speech.pth"
# LOAD_FROM_DISC = r"data\Coswara_processed\models\2023-07-06_epoch169_evalMetric_83.8_seed99468865_cough.pth"
# LOAD_FROM_DISC = r"data\Coswara_processed\models\2023-07-06_epoch91_evalMetric_83.7_seed99468865_vowels.pth"
LOAD_FROM_DISC = r"data\Coswara_processed\models\2023-07-11_epoch102_evalMetric_81.9_combined_breaths _seed99468865.pth"

# LOAD_FROM_DISC = r"data\Coswara_processed\models\2023-07-13_epoch19_evalMetric_85.2_combined_speech " \
#                  r"_seed99468865_trainedOnFullSet.pth "
# LOAD_FROM_DISC = r"data\Coswara_processed\models\2023-07-13_finalepoch95_evalMetric_80.5_combined_speech " \
#                  r"_seed99468865_trainedOnFullSet.pth "



# LOAD_FROM_DISC = r"data\Coswara_processed\models\2023-07-06_epoch169_evalMetric_83.8_seed99468865_cough.pth"
# LOAD_FROM_DISC = r"data\Coswara_processed\models\2023-07-06_epoch83_evalMetric_85.7_seed99468865_speech.pth"
# LOAD_FROM_DISC = r"data\Coswara_processed\models\2023-07-06_epoch91_evalMetric_83.7_seed99468865_vowels.pth"
# LOAD_FROM_DISC = r"data\Coswara_processed\models\2023-07-09_epoch149_evalMetric_84.9_seed99468865_vowels.pth"

# LOAD_FROM_DISC_multipleSplits = [
#     r"data\Coswara_processed\models\2023-07-11_epoch124_evalMetric_85.4_combined_speech _seed99468865.pth",
#     r"data\Coswara_processed\models\2023-07-11_epoch137_evalMetric_82.6_combined_speech _seed215674.pth",
#     r"data\Coswara_processed\models\2023-07-11_epoch136_evalMetric_83.8_combined_speech _seed3213213211.pth",
#     r"data\Coswara_processed\models\2023-07-11_epoch113_evalMetric_85.0_combined_speech _seed55555555.pth",
#     r"data\Coswara_processed\models\2023-07-11_epoch21_evalMetric_85.4_combined_speech _seed66445511337.pth",
# ]

# LOAD_FROM_DISC_multipleSplits = [
#     r"data\Coswara_processed\models\2023-07-11_epoch78_evalMetric_82.4_combined_coughs _seed99468865.pth",
#     r"data\Coswara_processed\models\2023-07-11_epoch11_evalMetric_82.6_combined_coughs _seed215674.pth",
#     r"data\Coswara_processed\models\2023-07-11_epoch76_evalMetric_83.8_combined_coughs _seed3213213211.pth",
#     r"data\Coswara_processed\models\2023-07-11_epoch52_evalMetric_84.8_combined_coughs _seed55555555.pth",
#     r"data\Coswara_processed\models\2023-07-11_epoch48_evalMetric_84.7_combined_coughs _seed66445511337.pth",
# ]

SAVE_TO_DISC = False
EVALUATE_TEST_SET = True
TRAIN_ON_FULL_SET = False

ID_PERFORMANCE_TRACKING = "test_linearRegression.pickle"

# ###########################################   DO NOT CHANGE LINES BELOW   ############################################
if isinstance(LOAD_FROM_DISC, str):
    LOAD_FROM_DISC = os.path.join(*LOAD_FROM_DISC.split("\\"))
if TRAIN_ON_FULL_SET:
    RUN_COMMENT += "_trainOnFullSet"
    EVALUATE_TEST_SET = False
# ID_PERFORMANCE_TRACKING = "data/Coswara_processed/id_performance_tracking/" + ID_PERFORMANCE_TRACKING
