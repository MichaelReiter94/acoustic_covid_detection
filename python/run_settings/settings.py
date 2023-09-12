# ########################################  MIL MIL MIL MIL MIL  #######################################################
# parameters = dict(
#     batch=[20],
#     lr=[1e-4, 1e-5, 1e-6],
#     lr_decay=[0.75],
#     lr_in=[None],
#     wd=[1e-4],
#
#     normalize=[True],
#     time_steps=[336],
#     use_augm_datasets=[False],
#     shift=[False],
#     sigma=[0],
#     mixup_a=[0],
#     mixup_p=[0],
#     dropout_p=[0],
#     transfer_func_sim=[0],
#     random_gain=[0],
#     exclude_outliers=[0],
#     use_resnorm=[False],
#     focal_loss=[0],
#
#     bag_size=[8, 10, 12],
#     lr_mil=[0.25, 1, 4],
#     n_MIL_Neurons=[32, 64],
#
#     exclude_conf_miscl=[False],
#     self_assessment_penalty=[0.8],
#     val_oversampl=[1],
#     class_weight=[1],
#     weighted_sampler=[False],
#     time_domain_augmentations_pos=[0],
#     time_domain_augmentations_neg=[0],
#     exclude_exposed=[True]  # no effect, just a reminder to include into the excel sheet
# )
# ########################################  MIL MIL MIL MIL MIL  #######################################################
# evaluation settings for saved models













#
# parameters = dict(
#     batch=[64],
#     lr=[1e-10],
#     lr_decay=[1],
#     lr_in=[None],
#     wd=[0],
#
#     normalize=[False],
#     time_steps=[336],
#     use_augm_datasets=[False],
#
#     shift=[True],
#     sigma=[0],
#     mixup_a=[0],
#     mixup_p=[0],
#     dropout_p=[0],
#     transfer_func_sim=[0],
#     random_gain=[0],
#
#     exclude_outliers=[0],
#     focal_loss=[0],
#
#     bag_size=[None],
#     lr_mil=[None],
#     n_MIL_Neurons=[None],
#
#     exclude_conf_miscl=[False],  # exclude confidently misclassified examples (manually identified and saved in excel
#     self_assessment_penalty=[1],
#
#     val_oversampl=[8],
#     class_weight=[1],
#     weighted_sampler=[True],
#     time_domain_augmentations_pos=[0],
#     time_domain_augmentations_neg=[0],
#     exclude_exposed=[True],  # no effect, just a reminder to include into the Excel sheet
#
#     use_resnorm=[False],
#     resnorm_affine=[True],
#     resnorm_gamma=[0.85],
#     input_resnorm=[True],
#     track_stats=[False],
#     dropout_p_MIL=[0.0]
#
# )
# # DATASET_NAME = "logmel_combined_breaths_NEW_23msHop_46msFFT_fmax11000_224logmel"
#
# # DATASET_NAME = "logmel_combined_breaths_NEW_46msHop_92msFFT_fmax11000_224logmel"
# # DATASET_NAME = "2023_05_22_logmel_combined_coughs_NEW_11msHop_23msFFT_fmax11000_224logmel"
# # DATASET_NAME = "2023_06_25_logmel_combined_speech_NEW_23msHop_46msFFT_fmax11000_224logmel"
# DATASET_NAME = "2023_06_25_logmel_combined_vowels_NEW_23msHop_96msFFT_fmax11000_224logmel"
#
#
# RUN_COMMENT = f"garbage"
# n_epochs = 1
# n_cross_validation_runs = 5
# USE_MIL = False
# MODEL_NAME = "resnet18"
# # MODEL_NAME = "resnet50"
#
# USE_TRAIN_VAL_TEST_SPLIT = True
# QUICK_TRAIN_FOR_TESTS = False
# SAMPLES_PER_EPOCH = 64
#
# LOAD_FROM_DISC = False
# # LOAD_FROM_DISC = r"2023-07-02_epoch104_evalMetric_77.9_seed99468865.pth"
# # LOAD_FROM_DISC = r"2023-07-02_epoch119_evalMetric_81.9_seed215674.pth"
# # LOAD_FROM_DISC = r"2023-07-06_epoch83_evalMetric_85.7_seed99468865_speech.pth"
# # LOAD_FROM_DISC = r"2023-07-06_epoch169_evalMetric_83.8_seed99468865_cough.pth"
# # LOAD_FROM_DISC = r"2023-07-06_epoch91_evalMetric_83.7_seed99468865_vowels.pth"
# # LOAD_FROM_DISC = r"2023-07-11_epoch102_evalMetric_81.9_combined_breaths _seed99468865.pth"
# # LOAD_FROM_DISC = r"2023-07-13_epoch19_evalMetric_85.2_combined_speech " \
# #                  r"_seed99468865_trainedOnFullSet.pth "
# # LOAD_FROM_DISC = r"2023-07-13_finalepoch95_evalMetric_80.5_combined_speech " \
# #                  r"_seed99468865_trainedOnFullSet.pth "
# # LOAD_FROM_DISC = r"2023-07-06_epoch169_evalMetric_83.8_seed99468865_cough.pth"
# # LOAD_FROM_DISC = r"2023-07-06_epoch83_evalMetric_85.7_seed99468865_speech.pth"
# # LOAD_FROM_DISC = r"2023-07-06_epoch91_evalMetric_83.7_seed99468865_vowels.pth"
# # LOAD_FROM_DISC = r"2023-07-09_epoch149_evalMetric_84.9_seed99468865_vowels.pth"
#
# # LOAD_FROM_DISC_multipleSplits = None
# # LOAD_FROM_DISC_multipleSplits = [
# #     r"2023-07-11_epoch124_evalMetric_85.4_combined_speech _seed99468865.pth",
# #     r"2023-07-11_epoch137_evalMetric_82.6_combined_speech _seed215674.pth",
# #     r"2023-07-11_epoch136_evalMetric_83.8_combined_speech _seed3213213211.pth",
# #     r"2023-07-11_epoch113_evalMetric_85.0_combined_speech _seed55555555.pth",
# #     r"2023-07-11_epoch21_evalMetric_85.4_combined_speech _seed66445511337.pth",
# # ]
# # LOAD_FROM_DISC_multipleSplits = [
# #     r"2023-07-11_epoch78_evalMetric_82.4_combined_coughs _seed99468865.pth",
# #     r"2023-07-11_epoch11_evalMetric_82.6_combined_coughs _seed215674.pth",
# #     r"2023-07-11_epoch76_evalMetric_83.8_combined_coughs _seed3213213211.pth",
# #     r"2023-07-11_epoch52_evalMetric_84.8_combined_coughs _seed55555555.pth",
# #     r"2023-07-11_epoch48_evalMetric_84.7_combined_coughs _seed66445511337.pth",
# # ]
#
# # #############################################     224 time steps    ##################################################
# # LOAD_FROM_DISC_multipleSplits = [
# #     "224_timesteps2023-08-15_epoch99_evalMetric_74.5_combined_breaths _seed99468865.pth",
# #     "224_timesteps2023-08-15_epoch87_evalMetric_79.5_combined_breaths _seed215674.pth",
# #     "224_timesteps2023-08-15_epoch119_evalMetric_76.8_combined_breaths _seed3213213211.pth",
# #     "224_timesteps2023-08-15_epoch46_evalMetric_79.1_combined_breaths _seed55555555.pth",
# #     "224_timesteps2023-08-15_epoch56_evalMetric_76.4_combined_breaths _seed66445511337.pth"
# # ]
# # LOAD_FROM_DISC_multipleSplits = [
# #     "224_timesteps2023-08-15_epoch68_evalMetric_81.6_combined_coughs _seed99468865.pth",
# #     "224_timesteps2023-08-15_epoch82_evalMetric_75.8_combined_coughs _seed215674.pth",
# #     "224_timesteps2023-08-15_epoch117_evalMetric_78.0_combined_coughs _seed3213213211.pth",
# #     "224_timesteps2023-08-15_epoch46_evalMetric_82.7_combined_coughs _seed55555555.pth",
# #     "224_timesteps2023-08-15_epoch15_evalMetric_77.2_combined_coughs _seed66445511337.pth"
# # ]
# # LOAD_FROM_DISC_multipleSplits = [
# #     "224_timesteps2023-08-15_epoch125_evalMetric_80.3_combined_speech _seed99468865.pth",
# #     "224_timesteps2023-08-15_epoch66_evalMetric_72.2_combined_speech _seed215674.pth",
# #     "224_timesteps2023-08-15_epoch87_evalMetric_75.5_combined_speech _seed3213213211.pth",
# #     "224_timesteps2023-08-15_epoch119_evalMetric_80.6_combined_speech _seed55555555.pth",
# #     "224_timesteps2023-08-15_epoch49_evalMetric_78.6_combined_speech _seed66445511337.pth"
# # ]
# # LOAD_FROM_DISC_multipleSplits = [
# #     "224_timesteps2023-08-15_epoch79_evalMetric_77.7_combined_vowels _seed99468865.pth",
# #     "224_timesteps2023-08-15_epoch104_evalMetric_74.7_combined_vowels _seed215674.pth",
# #     "224_timesteps2023-08-15_epoch83_evalMetric_76.3_combined_vowels _seed3213213211.pth",
# #     "224_timesteps2023-08-15_epoch50_evalMetric_82.4_combined_vowels _seed55555555.pth",
# #     "224_timesteps2023-08-15_epoch82_evalMetric_77.8_combined_vowels _seed66445511337.pth"
# # ]
# # #############################################     448 time steps    ##################################################
#
# # LOAD_FROM_DISC_multipleSplits = [
# #     "448_timesteps2023-08-15_epoch108_evalMetric_77.5_combined_breaths _seed99468865.pth",
# #     "448_timesteps2023-08-15_epoch61_evalMetric_80.0_combined_breaths _seed215674.pth",
# #     "448_timesteps2023-08-15_epoch117_evalMetric_76.8_combined_breaths _seed3213213211.pth",
# #     "448_timesteps2023-08-15_epoch108_evalMetric_79.4_combined_breaths _seed55555555.pth",
# #     "448_timesteps2023-08-15_epoch39_evalMetric_77.1_combined_breaths _seed66445511337.pth"
# # ]
#
# # LOAD_FROM_DISC_multipleSplits = [
# #     "448_timesteps2023-08-15_epoch115_evalMetric_81.9_combined_coughs _seed99468865.pth",
# #     "448_timesteps2023-08-15_epoch81_evalMetric_79.3_combined_coughs _seed215674.pth",
# #     "448_timesteps2023-08-15_epoch40_evalMetric_80.4_combined_coughs _seed3213213211.pth",
# #     "448_timesteps2023-08-15_epoch7_evalMetric_82.6_combined_coughs _seed55555555.pth",
# #     "448_timesteps2023-08-15_epoch92_evalMetric_78.9_combined_coughs _seed66445511337.pth"
# # ]
# # LOAD_FROM_DISC_multipleSplits = [
# #     "448_timesteps2023-08-15_epoch89_evalMetric_81.1_combined_speech _seed99468865.pth",
# #     "448_timesteps2023-08-15_epoch126_evalMetric_76.8_combined_speech _seed215674.pth",
# #     "448_timesteps2023-08-15_epoch45_evalMetric_78.2_combined_speech _seed3213213211.pth",
# #     "448_timesteps2023-08-15_epoch129_evalMetric_80.9_combined_speech _seed55555555.pth",
# #     "448_timesteps2023-08-15_epoch72_evalMetric_77.6_combined_speech _seed66445511337.pth"
# # ]
# # LOAD_FROM_DISC_multipleSplits = [
# #     "448_timesteps2023-08-15_epoch66_evalMetric_78.8_combined_vowels _seed99468865.pth",
# #     "448_timesteps2023-08-15_epoch120_evalMetric_75.5_combined_vowels _seed215674.pth",
# #     "448_timesteps2023-08-15_epoch31_evalMetric_77.3_combined_vowels _seed3213213211.pth",
# #     "448_timesteps2023-08-15_epoch18_evalMetric_81.6_combined_vowels _seed55555555.pth",
# #     "448_timesteps2023-08-15_epoch78_evalMetric_81.0_combined_vowels _seed66445511337.pth"
# # ]
#
# # ##################################     336 time steps with breath 26ms    ############################################
#
#
# # LOAD_FROM_DISC_multipleSplits = [
# #     "336_timesteps2023-08-25_epoch93_evalMetric_76.5_combined_breaths_26ms _seed99468865.pth",
# #     "336_timesteps2023-08-25_epoch30_evalMetric_77.8_combined_breaths_26ms _seed215674.pth",
# #     "336_timesteps2023-08-25_epoch103_evalMetric_75.7_combined_breaths_26ms _seed3213213211.pth",
# #     "336_timesteps2023-08-25_epoch59_evalMetric_79.6_combined_breaths_26ms _seed55555555.pth",
# #     "336_timesteps2023-08-25_epoch30_evalMetric_77.4_combined_breaths_26ms _seed66445511337.pth"
# # ]
#
#
# # ############################################    FINAL 336  steps     #################################################
#
#
# # LOAD_FROM_DISC_multipleSplits = [
# #     "336_timesteps2023-09-01_epoch196_evalMetric_77.1_combined_breaths _seed99468865.pth",
# #     "336_timesteps2023-09-01_epoch161_evalMetric_81.7_combined_breaths _seed215674.pth",
# #     "336_timesteps2023-09-01_epoch115_evalMetric_77.5_combined_breaths _seed3213213211.pth",
# #     "336_timesteps2023-09-01_epoch189_evalMetric_79.5_combined_breaths _seed55555555.pth",
# #     "336_timesteps2023-09-01_epoch112_evalMetric_76.7_combined_breaths _seed66445511337.pth"
# # ]

# ########################################  MIL MIL MIL MIL MIL  #######################################################
parameters = dict(
    batch=[6],
    lr=[1e-5, 3e-6],
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

    bag_size=[10],
    lr_mil=[0.25],
    n_MIL_Neurons=[256],

    exclude_conf_miscl=[True],
    self_assessment_penalty=[1.0],
    val_oversampl=[1],
    class_weight=[1],
    weighted_sampler=[False],
    time_domain_augmentations_pos=[0],
    time_domain_augmentations_neg=[0],
    exclude_exposed=[True],

    use_resnorm=[True],
    resnorm_affine=[True],
    resnorm_gamma=[0.85],
    input_resnorm=[True],
    track_stats=[False],
    dropout_p_MIL=[0.0]
)
# ########################################  MIL MIL MIL MIL MIL  #######################################################

USE_MIL = True

RUN_COMMENT = f"test_nwMetadataMIL"
n_epochs = 1
n_cross_validation_runs = 1

SAVE_TO_DISC = False
EVALUATE_TEST_SET = True
ID_PERFORMANCE_TRACKING = None

LOAD_FROM_DISC = False
# LOAD_FROM_DISC = "2023-06-12_resnet18_23ms_82_7_AUCROC_iter74.pth"
# LOAD_FROM_DISC = "2023-07-11_epoch102_evalMetric_81.9_combined_breaths _seed99468865.pth"

# LOAD_FROM_DISC_multipleSplits = [
#     "224_timesteps2023-08-15_epoch99_evalMetric_74.5_combined_breaths _seed99468865.pth",
#     "224_timesteps2023-08-15_epoch87_evalMetric_79.5_combined_breaths _seed215674.pth",
#     "224_timesteps2023-08-15_epoch119_evalMetric_76.8_combined_breaths _seed3213213211.pth",
#     "224_timesteps2023-08-15_epoch46_evalMetric_79.1_combined_breaths _seed55555555.pth",
#     "224_timesteps2023-08-15_epoch56_evalMetric_76.4_combined_breaths _seed66445511337.pth"
# ]
# LOAD_FROM_DISC_multipleSplits = [
#     "448_timesteps2023-08-15_epoch108_evalMetric_77.5_combined_breaths _seed99468865.pth",
#     "448_timesteps2023-08-15_epoch61_evalMetric_80.0_combined_breaths _seed215674.pth",
#     "448_timesteps2023-08-15_epoch117_evalMetric_76.8_combined_breaths _seed3213213211.pth",
#     "448_timesteps2023-08-15_epoch108_evalMetric_79.4_combined_breaths _seed55555555.pth",
#     "448_timesteps2023-08-15_epoch39_evalMetric_77.1_combined_breaths _seed66445511337.pth"
# ]

LOAD_FROM_DISC_multipleSplits = [
    "336_timesteps2023-09-01_epoch196_evalMetric_77.1_combined_breaths _seed99468865.pth",
    "336_timesteps2023-09-01_epoch161_evalMetric_81.7_combined_breaths _seed215674.pth",
    "336_timesteps2023-09-01_epoch115_evalMetric_77.5_combined_breaths _seed3213213211.pth",
    "336_timesteps2023-09-01_epoch189_evalMetric_79.5_combined_breaths _seed55555555.pth",
    "336_timesteps2023-09-01_epoch112_evalMetric_76.7_combined_breaths _seed66445511337.pth"
]

FREEZE_MODEL = False
# ###########################################   DO NOT CHANGE LINES BELOW   ############################################
DATASET_NAME = "logmel_combined_breaths_NEW_46msHop_92msFFT_fmax11000_224logmel"
# DATASET_NAME = "logmel_combined_breaths_NEW_23msHop_46msFFT_fmax11000_224logmel"

MODEL_NAME = "resnet18"
QUICK_TRAIN_FOR_TESTS = False
SAMPLES_PER_EPOCH = 1024
TRAIN_ON_FULL_SET = False
USE_TRAIN_VAL_TEST_SPLIT = True
