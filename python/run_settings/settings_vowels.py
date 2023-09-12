parameters = dict(
    batch=[128],
    lr=[3e-4],
    lr_decay=[0.99],
    lr_in=[1],
    wd=[1e-4],

    normalize=[False],
    time_steps=[336],
    use_augm_datasets=[True],

    shift=[True],
    sigma=[0],
    mixup_a=[0.4],
    mixup_p=[1],
    dropout_p=[0],
    transfer_func_sim=[6],
    random_gain=[3],

    exclude_outliers=[0],
    focal_loss=[2],

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
    exclude_exposed=[True],  # no effect, just a reminder to include into the Excel sheet

    use_resnorm=[False],
    resnorm_affine=[True],
    resnorm_gamma=[0.85],
    input_resnorm=[True],
    track_stats=[False],
    dropout_p_MIL=[0.0]

)

USE_MIL = False

RUN_COMMENT = f"resnorm_saveFinalModels"
n_epochs = 200
n_cross_validation_runs = 5

SAVE_TO_DISC = True
EVALUATE_TEST_SET = True
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
