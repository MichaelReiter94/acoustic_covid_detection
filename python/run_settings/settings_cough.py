parameters = dict(
    batch=[32],
    lr=[3e-4],
    lr_decay=[0.97],
    lr_in=[1],
    wd=[1e-4],

    normalize=[True],
    time_steps=[336],
    use_augm_datasets=[True],

    shift=[True],
    sigma=[0.01],
    mixup_a=[0.4],
    mixup_p=[1],
    dropout_p=[0],
    transfer_func_sim=[6],
    random_gain=[0],

    exclude_outliers=[0],
    focal_loss=[0.5],

    bag_size=[None],
    lr_mil=[None],
    n_MIL_Neurons=[None],

    exclude_conf_miscl=[True],
    self_assessment_penalty=[1],

    val_oversampl=[8],
    class_weight=[1],
    weighted_sampler=[True],

    time_domain_augmentations_pos=[0],
    time_domain_augmentations_neg=[1],
    exclude_exposed=[True],  # no effect, just a reminder to include into the Excel sheet

    use_resnorm=[False],
    resnorm_affine=[False],
    resnorm_gamma=[None],
    input_resnorm=[False],
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
# LOAD_FROM_DISC = r"data/Coswara_processed/models/2023-07-06_epoch169_evalMetric_83.8_seed99468865_cough.pth"
LOAD_FROM_DISC_multipleSplits = None
FREEZE_MODEL = False
# ###########################################   DO NOT CHANGE LINES BELOW   ############################################
DATASET_NAME = "2023_05_22_logmel_combined_coughs_NEW_11msHop_23msFFT_fmax11000_224logmel"
MODEL_NAME = "resnet18"
QUICK_TRAIN_FOR_TESTS = False
SAMPLES_PER_EPOCH = 1024
TRAIN_ON_FULL_SET = False
USE_TRAIN_VAL_TEST_SPLIT = True
