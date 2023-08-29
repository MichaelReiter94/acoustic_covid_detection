parameters = dict(
    batch=[128],
    lr=[1e-4, 6e-5],
    lr_decay=[0.99],
    lr_in=[0.3],
    wd=[1e-6],

    normalize=[True],
    time_steps=[336],
    use_augm_datasets=[True],

    shift=[True],
    sigma=[0.01],
    mixup_a=[0.2],
    mixup_p=[1],
    dropout_p=[0.15],
    transfer_func_sim=[1],
    random_gain=[0],

    exclude_outliers=[0],
    use_resnorm=[False],
    focal_loss=[0],

    bag_size=[None],
    lr_mil=[None],
    n_MIL_Neurons=[None],

    exclude_conf_miscl=[False],
    self_assessment_penalty=[0.8],

    val_oversampl=[8],
    class_weight=[1],
    weighted_sampler=[True],

    time_domain_augmentations_pos=[0, 2, 4],
    time_domain_augmentations_neg=[0, 1],
    exclude_exposed=[True]  # no effect, just a reminder to include into the excel sheet
)

USE_MIL = False

RUN_COMMENT = f"time_domain_augments"
n_epochs = 130
n_cross_validation_runs = 3

SAVE_TO_DISC = False
EVALUATE_TEST_SET = False
ID_PERFORMANCE_TRACKING = None

LOAD_FROM_DISC = False
# LOAD_FROM_DISC = r"data\Coswara_processed\models\2023-06-12_resnet18_23ms_82_7_AUCROC_iter74.pth"
# LOAD_FROM_DISC = r"data\Coswara_processed\models\2023-07-11_epoch102_evalMetric_81.9_combined_breaths _seed99468865.pth"
LOAD_FROM_DISC_multipleSplits = None
FREEZE_MODEL = False
# ###########################################   DO NOT CHANGE LINES BELOW   ############################################
# DATASET_NAME = "logmel_combined_breaths_NEW_46msHop_92msFFT_fmax11000_224logmel"
DATASET_NAME = "logmel_combined_breaths_NEW_23msHop_46msFFT_fmax11000_224logmel"
# DATASET_NAME = "logmel_combined_breaths_ALTERNATIVE_RES_46msHop_92msFFT_fmax5500"



MODEL_NAME = "resnet18"
QUICK_TRAIN_FOR_TESTS = False
SAMPLES_PER_EPOCH = 1024
TRAIN_ON_FULL_SET = False
USE_TRAIN_VAL_TEST_SPLIT = True
