
# ########################################  MIL MIL MIL MIL MIL  #######################################################
parameters = dict(
    batch=[10],
    lr=[1e-5],
    lr_decay=[0.2],
    lr_in=[0.2],
    wd=[1e-5],

    normalize=[True],
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
    focal_loss=[0],

    bag_size=[12],
    lr_mil=[0.25],
    n_MIL_Neurons=[64],

    exclude_conf_miscl=[True],
    self_assessment_penalty=[1.0],
    val_oversampl=[1],
    class_weight=[1],
    weighted_sampler=[False],
    time_domain_augmentations_pos=[0],
    time_domain_augmentations_neg=[0],
    exclude_exposed=[True],

    use_resnorm=[False],
    resnorm_affine=[False],
    resnorm_gamma=[None],
    input_resnorm=[False],
    track_stats=[False],
    dropout_p_MIL=[0.0]
)
# #########################################  MIL MIL MIL MIL MIL  ######################################################

USE_MIL = True

RUN_COMMENT = f"metadataMIL_saveIDPerformance"
n_epochs = 2
n_cross_validation_runs = 5

SAVE_TO_DISC = False
EVALUATE_TEST_SET = True
ID_PERFORMANCE_TRACKING = "metadata_mil.pickle"

LOAD_FROM_DISC = False
# LOAD_FROM_DISC = "2023-07-06_epoch83_evalMetric_85.7_seed99468865_speech.pth"
# LOAD_FROM_DISC_multipleSplits = [
#     "2023-07-11_epoch124_evalMetric_85.4_combined_speech _seed99468865.pth",
#     "2023-07-11_epoch137_evalMetric_82.6_combined_speech _seed215674.pth",
#     "2023-07-11_epoch136_evalMetric_83.8_combined_speech _seed3213213211.pth",
#     "2023-07-11_epoch113_evalMetric_85.0_combined_speech _seed55555555.pth",
#     "2023-07-11_epoch21_evalMetric_85.4_combined_speech _seed66445511337.pth"]
# LOAD_FROM_DISC_multipleSplits = [
#     "224_timesteps2023-08-15_epoch125_evalMetric_80.3_combined_speech _seed99468865.pth",
#     "224_timesteps2023-08-15_epoch66_evalMetric_72.2_combined_speech _seed215674.pth",
#     "224_timesteps2023-08-15_epoch87_evalMetric_75.5_combined_speech _seed3213213211.pth",
#     "224_timesteps2023-08-15_epoch119_evalMetric_80.6_combined_speech _seed55555555.pth",
#     "224_timesteps2023-08-15_epoch49_evalMetric_78.6_combined_speech _seed66445511337.pth"
# ]

# LOAD_FROM_DISC_multipleSplits = [
#     "448_timesteps2023-08-15_epoch89_evalMetric_81.1_combined_speech _seed99468865.pth",
#     "448_timesteps2023-08-15_epoch126_evalMetric_76.8_combined_speech _seed215674.pth",
#     "448_timesteps2023-08-15_epoch45_evalMetric_78.2_combined_speech _seed3213213211.pth",
#     "448_timesteps2023-08-15_epoch129_evalMetric_80.9_combined_speech _seed55555555.pth",
#     "448_timesteps2023-08-15_epoch72_evalMetric_77.6_combined_speech _seed66445511337.pth"
# ]

LOAD_FROM_DISC_multipleSplits = [
    "336_timesteps2023-09-01_epoch195_evalMetric_82.6_combined_speech _seed99468865.pth",
    "336_timesteps2023-09-01_epoch148_evalMetric_79.5_combined_speech _seed215674.pth",
    "336_timesteps2023-09-01_epoch78_evalMetric_77.7_combined_speech _seed3213213211.pth",
    "336_timesteps2023-09-01_epoch127_evalMetric_82.5_combined_speech _seed55555555.pth",
    "336_timesteps2023-09-01_epoch13_evalMetric_78.3_combined_speech _seed66445511337.pth"
]

FREEZE_MODEL = False
# ###########################################   DO NOT CHANGE LINES BELOW   ############################################
DATASET_NAME = "2023_06_25_logmel_combined_speech_NEW_23msHop_46msFFT_fmax11000_224logmel"
MODEL_NAME = "resnet18"
QUICK_TRAIN_FOR_TESTS = False
SAMPLES_PER_EPOCH = 1024
TRAIN_ON_FULL_SET = False
USE_TRAIN_VAL_TEST_SPLIT = True

