parameters = dict(
    batch=[128],
    lr=[6e-5],
    lr_decay=[0.99],
    lr_in=[0.6],
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
    focal_loss=[1],

    bag_size=[None],
    lr_mil=[None],
    n_MIL_Neurons=[None],

    exclude_conf_miscl=[True],
    self_assessment_penalty=[0.8],

    val_oversampl=[8],
    class_weight=[1],
    weighted_sampler=[True],

    time_domain_augmentations_pos=[2],
    time_domain_augmentations_neg=[1],
    exclude_exposed=[True],  # no effect, just a reminder to include into the Excel sheet

    use_resnorm=[True],
    resnorm_affine=[True],
    resnorm_gamma=[0.85],
    input_resnorm=[True],
    track_stats=[True],
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
# LOAD_FROM_DISC = r"data\Coswara_processed\models\2023-06-12_resnet18_23ms_82_7_AUCROC_iter74.pth"
# LOAD_FROM_DISC = r"data\Coswara_processed\models\2023-07-11_epoch102_evalMetric_81.9_combined_breaths _seed99468865.pth"
LOAD_FROM_DISC_multipleSplits = None
FREEZE_MODEL = False
# ###########################################   DO NOT CHANGE LINES BELOW   ############################################
DATASET_NAME = "logmel_combined_breaths_NEW_46msHop_92msFFT_fmax11000_224logmel"
MODEL_NAME = "resnet18"
QUICK_TRAIN_FOR_TESTS = False
SAMPLES_PER_EPOCH = 1024
TRAIN_ON_FULL_SET = False
USE_TRAIN_VAL_TEST_SPLIT = True
