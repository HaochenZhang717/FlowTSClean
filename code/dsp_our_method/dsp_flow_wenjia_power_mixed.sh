cd ..

export hucfg_t_sampling=logitnorm
LR=1e-4
LEN_WHOLE=384
MAX_LEN_ANOMALY=144
MIN_LEN_ANOMALY=144

ONE_CHANNEL=1
FEAT_SIZE=1

DATA_TYPE="ecg"
WANDB_PROJECT="dsp_flow_wenjia_power"


#DATA_PATHS='["./dataset_utils/ECG_datasets/raw_data_PV/2013_pv_sub_2.npz","./dataset_utils/ECG_datasets/raw_data_PV/2013_pv_sub_3.npz","./dataset_utils/ECG_datasets/raw_data_PV/2013_pv_sub_4.npz","./dataset_utils/ECG_datasets/raw_data_PV/2015_pv_sub_0.npz","./dataset_utils/ECG_datasets/raw_data_PV/2015_pv_sub_1.npz","./dataset_utils/ECG_datasets/raw_data_PV/2021_pv_live_0.npz","./dataset_utils/ECG_datasets/raw_data_PV/2022_pv_live_0.npz","./dataset_utils/ECG_datasets/raw_data_PV/2023_pv_live_0.npz","./dataset_utils/ECG_datasets/raw_data_PV/2024_pv_live_0.npz","./dataset_utils/ECG_datasets/raw_data_PV/2025_pv_live_0.npz"]'
DATA_PATHS='["../data_set_processing/npz_files/NYISO_CAPITL_load_anomaly_2024_30min.npz","../data_set_processing/npz_files/NYISO_CENTRL_load_anomaly_2024_30min.npz","../data_set_processing/npz_files/NYISO_DUNWOD_load_anomaly_2024_30min.npz","../data_set_processing/npz_files/NYISO_GENESE_load_anomaly_2024_30min.npz","../data_set_processing/npz_files/NYISO_HUD_VL_load_anomaly_2024_30min.npz","../data_set_processing/npz_files/NYISO_LONGIL_load_anomaly_2024_30min.npz","../data_set_processing/npz_files/NYISO_MHK_VL_load_anomaly_2024_30min.npz","../data_set_processing/npz_files/NYISO_MILLWD_load_anomaly_2024_30min.npz","../data_set_processing/npz_files/NYISO_NORTH_load_anomaly_2024_30min.npz","../data_set_processing/npz_files/NYISO_N_Y_C_load_anomaly_2024_30min.npz","../data_set_processing/npz_files/NYISO_WEST_load_anomaly_2024_30min.npz","../data_set_processing/npz_files/PJM_MIDATL_load_anomaly_2024_30min.npz","../data_set_processing/npz_files/PJM_RTO_load_anomaly_2024_30min.npz","../data_set_processing/npz_files/PJM_SOUTH_load_anomaly_2024_30min.npz","../data_set_processing/npz_files/PJM_WEST_load_anomaly_2024_30min.npz","../data_set_processing/npz_files/SPP_CSWS_load_anomaly_2021_30min.npz","../data_set_processing/npz_files/SPP_EDE_load_anomaly_2021_30min.npz","../data_set_processing/npz_files/SPP_GRDA_load_anomaly_2021_30min.npz","../data_set_processing/npz_files/SPP_INDN_load_anomaly_2021_30min.npz","../data_set_processing/npz_files/SPP_KACY_load_anomaly_2021_30min.npz","../data_set_processing/npz_files/SPP_KCPL_load_anomaly_2021_30min.npz","../data_set_processing/npz_files/SPP_LES_load_anomaly_2021_30min.npz","../data_set_processing/npz_files/SPP_MPS_load_anomaly_2021_30min.npz","../data_set_processing/npz_files/SPP_NPPD_load_anomaly_2021_30min.npz","../data_set_processing/npz_files/SPP_OKGE_load_anomaly_2021_30min.npz","../data_set_processing/npz_files/SPP_OPPD_load_anomaly_2021_30min.npz","../data_set_processing/npz_files/SPP_SECI_load_anomaly_2021_30min.npz","../data_set_processing/npz_files/SPP_SPRM_load_anomaly_2021_30min.npz","../data_set_processing/npz_files/SPP_SPS_load_anomaly_2021_30min.npz","../data_set_processing/npz_files/SPP_WAUE_load_anomaly_2021_30min.npz","../data_set_processing/npz_files/SPP_WFEC_load_anomaly_2021_30min.npz","../data_set_processing/npz_files/SPP_WR_load_anomaly_2021_30min.npz"]'

#TEST_DATA_PATHS='["./dataset_utils/ECG_datasets/raw_data_PV/2013_pv_sub_2.npz","./dataset_utils/ECG_datasets/raw_data_PV/2013_pv_sub_3.npz","./dataset_utils/ECG_datasets/raw_data_PV/2013_pv_sub_4.npz","./dataset_utils/ECG_datasets/raw_data_PV/2015_pv_sub_0.npz","./dataset_utils/ECG_datasets/raw_data_PV/2015_pv_sub_1.npz","./dataset_utils/ECG_datasets/raw_data_PV/2021_pv_live_0.npz","./dataset_utils/ECG_datasets/raw_data_PV/2022_pv_live_0.npz","./dataset_utils/ECG_datasets/raw_data_PV/2023_pv_live_0.npz","./dataset_utils/ECG_datasets/raw_data_PV/2024_pv_live_0.npz","./dataset_utils/ECG_datasets/raw_data_PV/2025_pv_live_0.npz"]'
TEST_DATA_PATHS='["../data_set_processing/npz_files/NYISO_CAPITL_load_anomaly_2024_30min.npz","../data_set_processing/npz_files/NYISO_CENTRL_load_anomaly_2024_30min.npz","../data_set_processing/npz_files/NYISO_DUNWOD_load_anomaly_2024_30min.npz","../data_set_processing/npz_files/NYISO_GENESE_load_anomaly_2024_30min.npz","../data_set_processing/npz_files/NYISO_HUD_VL_load_anomaly_2024_30min.npz","../data_set_processing/npz_files/NYISO_LONGIL_load_anomaly_2024_30min.npz","../data_set_processing/npz_files/NYISO_MHK_VL_load_anomaly_2024_30min.npz","../data_set_processing/npz_files/NYISO_MILLWD_load_anomaly_2024_30min.npz","../data_set_processing/npz_files/NYISO_NORTH_load_anomaly_2024_30min.npz","../data_set_processing/npz_files/NYISO_N_Y_C_load_anomaly_2024_30min.npz","../data_set_processing/npz_files/NYISO_WEST_load_anomaly_2024_30min.npz","../data_set_processing/npz_files/PJM_MIDATL_load_anomaly_2024_30min.npz","../data_set_processing/npz_files/PJM_RTO_load_anomaly_2024_30min.npz","../data_set_processing/npz_files/PJM_SOUTH_load_anomaly_2024_30min.npz","../data_set_processing/npz_files/PJM_WEST_load_anomaly_2024_30min.npz","../data_set_processing/npz_files/SPP_CSWS_load_anomaly_2021_30min.npz","../data_set_processing/npz_files/SPP_EDE_load_anomaly_2021_30min.npz","../data_set_processing/npz_files/SPP_GRDA_load_anomaly_2021_30min.npz","../data_set_processing/npz_files/SPP_INDN_load_anomaly_2021_30min.npz","../data_set_processing/npz_files/SPP_KACY_load_anomaly_2021_30min.npz","../data_set_processing/npz_files/SPP_KCPL_load_anomaly_2021_30min.npz","../data_set_processing/npz_files/SPP_LES_load_anomaly_2021_30min.npz","../data_set_processing/npz_files/SPP_MPS_load_anomaly_2021_30min.npz","../data_set_processing/npz_files/SPP_NPPD_load_anomaly_2021_30min.npz","../data_set_processing/npz_files/SPP_OKGE_load_anomaly_2021_30min.npz","../data_set_processing/npz_files/SPP_OPPD_load_anomaly_2021_30min.npz","../data_set_processing/npz_files/SPP_SECI_load_anomaly_2021_30min.npz","../data_set_processing/npz_files/SPP_SPRM_load_anomaly_2021_30min.npz","../data_set_processing/npz_files/SPP_SPS_load_anomaly_2021_30min.npz","../data_set_processing/npz_files/SPP_WAUE_load_anomaly_2021_30min.npz","../data_set_processing/npz_files/SPP_WFEC_load_anomaly_2021_30min.npz","../data_set_processing/npz_files/SPP_WR_load_anomaly_2021_30min.npz"]'

#PRETRAIN_INDICES_PATHS='["./dataset_utils/ECG_datasets/indices_PV/slide_windows_2013_pv_sub_2npz/mixed.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2013_pv_sub_3npz/mixed.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2013_pv_sub_4npz/mixed.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2015_pv_sub_0npz/mixed.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2015_pv_sub_1npz/mixed.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2021_pv_live_0npz/mixed.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2022_pv_live_0npz/mixed.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2023_pv_live_0npz/mixed.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2024_pv_live_0npz/mixed.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2025_pv_live_0npz/mixed.jsonl"]'
PRETRAIN_INDICES_PATHS='["../data_set_processing/mixed_indices/NYISO_CAPITL_load_anomaly_2024_30min/mixed.jsonl","../data_set_processing/mixed_indices/NYISO_CENTRL_load_anomaly_2024_30min/mixed.jsonl","../data_set_processing/mixed_indices/NYISO_DUNWOD_load_anomaly_2024_30min/mixed.jsonl","../data_set_processing/mixed_indices/NYISO_GENESE_load_anomaly_2024_30min/mixed.jsonl","../data_set_processing/mixed_indices/NYISO_HUD_VL_load_anomaly_2024_30min/mixed.jsonl","../data_set_processing/mixed_indices/NYISO_LONGIL_load_anomaly_2024_30min/mixed.jsonl","../data_set_processing/mixed_indices/NYISO_MHK_VL_load_anomaly_2024_30min/mixed.jsonl","../data_set_processing/mixed_indices/NYISO_MILLWD_load_anomaly_2024_30min/mixed.jsonl","../data_set_processing/mixed_indices/NYISO_NORTH_load_anomaly_2024_30min/mixed.jsonl","../data_set_processing/mixed_indices/NYISO_N_Y_C_load_anomaly_2024_30min/mixed.jsonl","../data_set_processing/mixed_indices/NYISO_WEST_load_anomaly_2024_30min/mixed.jsonl","../data_set_processing/mixed_indices/PJM_MIDATL_load_anomaly_2024_30min/mixed.jsonl","../data_set_processing/mixed_indices/PJM_RTO_load_anomaly_2024_30min/mixed.jsonl","../data_set_processing/mixed_indices/PJM_SOUTH_load_anomaly_2024_30min/mixed.jsonl","../data_set_processing/mixed_indices/PJM_WEST_load_anomaly_2024_30min/mixed.jsonl","../data_set_processing/mixed_indices/SPP_CSWS_load_anomaly_2021_30min/mixed.jsonl","../data_set_processing/mixed_indices/SPP_EDE_load_anomaly_2021_30min/mixed.jsonl","../data_set_processing/mixed_indices/SPP_GRDA_load_anomaly_2021_30min/mixed.jsonl","../data_set_processing/mixed_indices/SPP_INDN_load_anomaly_2021_30min/mixed.jsonl","../data_set_processing/mixed_indices/SPP_KACY_load_anomaly_2021_30min/mixed.jsonl","../data_set_processing/mixed_indices/SPP_KCPL_load_anomaly_2021_30min/mixed.jsonl","../data_set_processing/mixed_indices/SPP_LES_load_anomaly_2021_30min/mixed.jsonl","../data_set_processing/mixed_indices/SPP_MPS_load_anomaly_2021_30min/mixed.jsonl","../data_set_processing/mixed_indices/SPP_NPPD_load_anomaly_2021_30min/mixed.jsonl","../data_set_processing/mixed_indices/SPP_OKGE_load_anomaly_2021_30min/mixed.jsonl","../data_set_processing/mixed_indices/SPP_OPPD_load_anomaly_2021_30min/mixed.jsonl","../data_set_processing/mixed_indices/SPP_SECI_load_anomaly_2021_30min/mixed.jsonl","../data_set_processing/mixed_indices/SPP_SPRM_load_anomaly_2021_30min/mixed.jsonl","../data_set_processing/mixed_indices/SPP_SPS_load_anomaly_2021_30min/mixed.jsonl","../data_set_processing/mixed_indices/SPP_WAUE_load_anomaly_2021_30min/mixed.jsonl","../data_set_processing/mixed_indices/SPP_WFEC_load_anomaly_2021_30min/mixed.jsonl","../data_set_processing/mixed_indices/SPP_WR_load_anomaly_2021_30min/mixed.jsonl"]'


#FINETUNE_TRAIN_INDICES_PATHS='["./dataset_utils/ECG_datasets/indices_PV/slide_windows_2013_pv_sub_2npz/V_train.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2013_pv_sub_3npz/V_train.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2013_pv_sub_4npz/V_train.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2015_pv_sub_0npz/V_train.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2015_pv_sub_1npz/V_train.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2021_pv_live_0npz/V_train.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2022_pv_live_0npz/V_train.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2023_pv_live_0npz/V_train.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2024_pv_live_0npz/V_train.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2025_pv_live_0npz/V_train.jsonl"]'
FINETUNE_TRAIN_INDICES_PATHS='["../data_set_processing/ts_with_anomaly_indices/NYISO_CAPITL_load_anomaly_2024_30min/V.jsonl","../data_set_processing/ts_with_anomaly_indices/NYISO_CENTRL_load_anomaly_2024_30min/V.jsonl","../data_set_processing/ts_with_anomaly_indices/NYISO_DUNWOD_load_anomaly_2024_30min/V.jsonl","../data_set_processing/ts_with_anomaly_indices/NYISO_GENESE_load_anomaly_2024_30min/V.jsonl","../data_set_processing/ts_with_anomaly_indices/NYISO_HUD_VL_load_anomaly_2024_30min/V.jsonl","../data_set_processing/ts_with_anomaly_indices/NYISO_LONGIL_load_anomaly_2024_30min/V.jsonl","../data_set_processing/ts_with_anomaly_indices/NYISO_MHK_VL_load_anomaly_2024_30min/V.jsonl","../data_set_processing/ts_with_anomaly_indices/NYISO_MILLWD_load_anomaly_2024_30min/V.jsonl","../data_set_processing/ts_with_anomaly_indices/NYISO_NORTH_load_anomaly_2024_30min/V.jsonl","../data_set_processing/ts_with_anomaly_indices/NYISO_N_Y_C_load_anomaly_2024_30min/V.jsonl","../data_set_processing/ts_with_anomaly_indices/NYISO_WEST_load_anomaly_2024_30min/V.jsonl","../data_set_processing/ts_with_anomaly_indices/PJM_MIDATL_load_anomaly_2024_30min/V.jsonl","../data_set_processing/ts_with_anomaly_indices/PJM_RTO_load_anomaly_2024_30min/V.jsonl","../data_set_processing/ts_with_anomaly_indices/PJM_SOUTH_load_anomaly_2024_30min/V.jsonl","../data_set_processing/ts_with_anomaly_indices/PJM_WEST_load_anomaly_2024_30min/V.jsonl","../data_set_processing/ts_with_anomaly_indices/SPP_CSWS_load_anomaly_2021_30min/V.jsonl","../data_set_processing/ts_with_anomaly_indices/SPP_EDE_load_anomaly_2021_30min/V.jsonl","../data_set_processing/ts_with_anomaly_indices/SPP_GRDA_load_anomaly_2021_30min/V.jsonl","../data_set_processing/ts_with_anomaly_indices/SPP_INDN_load_anomaly_2021_30min/V.jsonl","../data_set_processing/ts_with_anomaly_indices/SPP_KACY_load_anomaly_2021_30min/V.jsonl","../data_set_processing/ts_with_anomaly_indices/SPP_KCPL_load_anomaly_2021_30min/V.jsonl","../data_set_processing/ts_with_anomaly_indices/SPP_LES_load_anomaly_2021_30min/V.jsonl","../data_set_processing/ts_with_anomaly_indices/SPP_MPS_load_anomaly_2021_30min/V.jsonl","../data_set_processing/ts_with_anomaly_indices/SPP_NPPD_load_anomaly_2021_30min/V.jsonl","../data_set_processing/ts_with_anomaly_indices/SPP_OKGE_load_anomaly_2021_30min/V.jsonl","../data_set_processing/ts_with_anomaly_indices/SPP_OPPD_load_anomaly_2021_30min/V.jsonl","../data_set_processing/ts_with_anomaly_indices/SPP_SECI_load_anomaly_2021_30min/V.jsonl","../data_set_processing/ts_with_anomaly_indices/SPP_SPRM_load_anomaly_2021_30min/V.jsonl","../data_set_processing/ts_with_anomaly_indices/SPP_SPS_load_anomaly_2021_30min/V.jsonl","../data_set_processing/ts_with_anomaly_indices/SPP_WAUE_load_anomaly_2021_30min/V.jsonl","../data_set_processing/ts_with_anomaly_indices/SPP_WFEC_load_anomaly_2021_30min/V.jsonl","../data_set_processing/ts_with_anomaly_indices/SPP_WR_load_anomaly_2021_30min/V.jsonl"]'

#FINETUNE_TEST_INDICES_PATHS='["./dataset_utils/ECG_datasets/indices_PV/slide_windows_2013_pv_sub_2npz/V_test.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2013_pv_sub_3npz/V_test.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2013_pv_sub_4npz/V_test.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2015_pv_sub_0npz/V_test.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2015_pv_sub_1npz/V_test.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2021_pv_live_0npz/V_test.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2022_pv_live_0npz/V_test.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2023_pv_live_0npz/V_test.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2024_pv_live_0npz/V_test.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2025_pv_live_0npz/V_test.jsonl"]'
FINETUNE_TEST_INDICES_PATHS='["../data_set_processing/ts_with_anomaly_indices/NYISO_CAPITL_load_anomaly_2024_30min/V.jsonl","../data_set_processing/ts_with_anomaly_indices/NYISO_CENTRL_load_anomaly_2024_30min/V.jsonl","../data_set_processing/ts_with_anomaly_indices/NYISO_DUNWOD_load_anomaly_2024_30min/V.jsonl","../data_set_processing/ts_with_anomaly_indices/NYISO_GENESE_load_anomaly_2024_30min/V.jsonl","../data_set_processing/ts_with_anomaly_indices/NYISO_HUD_VL_load_anomaly_2024_30min/V.jsonl","../data_set_processing/ts_with_anomaly_indices/NYISO_LONGIL_load_anomaly_2024_30min/V.jsonl","../data_set_processing/ts_with_anomaly_indices/NYISO_MHK_VL_load_anomaly_2024_30min/V.jsonl","../data_set_processing/ts_with_anomaly_indices/NYISO_MILLWD_load_anomaly_2024_30min/V.jsonl","../data_set_processing/ts_with_anomaly_indices/NYISO_NORTH_load_anomaly_2024_30min/V.jsonl","../data_set_processing/ts_with_anomaly_indices/NYISO_N_Y_C_load_anomaly_2024_30min/V.jsonl","../data_set_processing/ts_with_anomaly_indices/NYISO_WEST_load_anomaly_2024_30min/V.jsonl","../data_set_processing/ts_with_anomaly_indices/PJM_MIDATL_load_anomaly_2024_30min/V.jsonl","../data_set_processing/ts_with_anomaly_indices/PJM_RTO_load_anomaly_2024_30min/V.jsonl","../data_set_processing/ts_with_anomaly_indices/PJM_SOUTH_load_anomaly_2024_30min/V.jsonl","../data_set_processing/ts_with_anomaly_indices/PJM_WEST_load_anomaly_2024_30min/V.jsonl","../data_set_processing/ts_with_anomaly_indices/SPP_CSWS_load_anomaly_2021_30min/V.jsonl","../data_set_processing/ts_with_anomaly_indices/SPP_EDE_load_anomaly_2021_30min/V.jsonl","../data_set_processing/ts_with_anomaly_indices/SPP_GRDA_load_anomaly_2021_30min/V.jsonl","../data_set_processing/ts_with_anomaly_indices/SPP_INDN_load_anomaly_2021_30min/V.jsonl","../data_set_processing/ts_with_anomaly_indices/SPP_KACY_load_anomaly_2021_30min/V.jsonl","../data_set_processing/ts_with_anomaly_indices/SPP_KCPL_load_anomaly_2021_30min/V.jsonl","../data_set_processing/ts_with_anomaly_indices/SPP_LES_load_anomaly_2021_30min/V.jsonl","../data_set_processing/ts_with_anomaly_indices/SPP_MPS_load_anomaly_2021_30min/V.jsonl","../data_set_processing/ts_with_anomaly_indices/SPP_NPPD_load_anomaly_2021_30min/V.jsonl","../data_set_processing/ts_with_anomaly_indices/SPP_OKGE_load_anomaly_2021_30min/V.jsonl","../data_set_processing/ts_with_anomaly_indices/SPP_OPPD_load_anomaly_2021_30min/V.jsonl","../data_set_processing/ts_with_anomaly_indices/SPP_SECI_load_anomaly_2021_30min/V.jsonl","../data_set_processing/ts_with_anomaly_indices/SPP_SPRM_load_anomaly_2021_30min/V.jsonl","../data_set_processing/ts_with_anomaly_indices/SPP_SPS_load_anomaly_2021_30min/V.jsonl","../data_set_processing/ts_with_anomaly_indices/SPP_WAUE_load_anomaly_2021_30min/V.jsonl","../data_set_processing/ts_with_anomaly_indices/SPP_WFEC_load_anomaly_2021_30min/V.jsonl","../data_set_processing/ts_with_anomaly_indices/SPP_WR_load_anomaly_2021_30min/V.jsonl"]'

#ANOMALY_INDICES_FOR_SAMPLE='["./dataset_utils/ECG_datasets/indices_PV/slide_windows_2013_pv_sub_2npz/V_segments_train.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2013_pv_sub_3npz/V_segments_train.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2013_pv_sub_4npz/V_segments_train.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2015_pv_sub_0npz/V_segments_train.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2015_pv_sub_1npz/V_segments_train.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2021_pv_live_0npz/V_segments_train.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2022_pv_live_0npz/V_segments_train.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2023_pv_live_0npz/V_segments_train.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2024_pv_live_0npz/V_segments_train.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2025_pv_live_0npz/V_segments_train.jsonl"]'
ANOMALY_INDICES_FOR_SAMPLE='["../data_set_processing/anomaly_indices/NYISO_CAPITL_load_anomaly_2024_30min/V_segments.jsonl","../data_set_processing/anomaly_indices/NYISO_CENTRL_load_anomaly_2024_30min/V_segments.jsonl","../data_set_processing/anomaly_indices/NYISO_DUNWOD_load_anomaly_2024_30min/V_segments.jsonl","../data_set_processing/anomaly_indices/NYISO_GENESE_load_anomaly_2024_30min/V_segments.jsonl","../data_set_processing/anomaly_indices/NYISO_HUD_VL_load_anomaly_2024_30min/V_segments.jsonl","../data_set_processing/anomaly_indices/NYISO_LONGIL_load_anomaly_2024_30min/V_segments.jsonl","../data_set_processing/anomaly_indices/NYISO_MHK_VL_load_anomaly_2024_30min/V_segments.jsonl","../data_set_processing/anomaly_indices/NYISO_MILLWD_load_anomaly_2024_30min/V_segments.jsonl","../data_set_processing/anomaly_indices/NYISO_NORTH_load_anomaly_2024_30min/V_segments.jsonl","../data_set_processing/anomaly_indices/NYISO_N_Y_C_load_anomaly_2024_30min/V_segments.jsonl","../data_set_processing/anomaly_indices/NYISO_WEST_load_anomaly_2024_30min/V_segments.jsonl","../data_set_processing/anomaly_indices/PJM_MIDATL_load_anomaly_2024_30min/V_segments.jsonl","../data_set_processing/anomaly_indices/PJM_RTO_load_anomaly_2024_30min/V_segments.jsonl","../data_set_processing/anomaly_indices/PJM_SOUTH_load_anomaly_2024_30min/V_segments.jsonl","../data_set_processing/anomaly_indices/PJM_WEST_load_anomaly_2024_30min/V_segments.jsonl","../data_set_processing/anomaly_indices/SPP_CSWS_load_anomaly_2021_30min/V_segments.jsonl","../data_set_processing/anomaly_indices/SPP_EDE_load_anomaly_2021_30min/V_segments.jsonl","../data_set_processing/anomaly_indices/SPP_GRDA_load_anomaly_2021_30min/V_segments.jsonl","../data_set_processing/anomaly_indices/SPP_INDN_load_anomaly_2021_30min/V_segments.jsonl","../data_set_processing/anomaly_indices/SPP_KACY_load_anomaly_2021_30min/V_segments.jsonl","../data_set_processing/anomaly_indices/SPP_KCPL_load_anomaly_2021_30min/V_segments.jsonl","../data_set_processing/anomaly_indices/SPP_LES_load_anomaly_2021_30min/V_segments.jsonl","../data_set_processing/anomaly_indices/SPP_MPS_load_anomaly_2021_30min/V_segments.jsonl","../data_set_processing/anomaly_indices/SPP_NPPD_load_anomaly_2021_30min/V_segments.jsonl","../data_set_processing/anomaly_indices/SPP_OKGE_load_anomaly_2021_30min/V_segments.jsonl","../data_set_processing/anomaly_indices/SPP_OPPD_load_anomaly_2021_30min/V_segments.jsonl","../data_set_processing/anomaly_indices/SPP_SECI_load_anomaly_2021_30min/V_segments.jsonl","../data_set_processing/anomaly_indices/SPP_SPRM_load_anomaly_2021_30min/V_segments.jsonl","../data_set_processing/anomaly_indices/SPP_SPS_load_anomaly_2021_30min/V_segments.jsonl","../data_set_processing/anomaly_indices/SPP_WAUE_load_anomaly_2021_30min/V_segments.jsonl","../data_set_processing/anomaly_indices/SPP_WFEC_load_anomaly_2021_30min/V_segments.jsonl","../data_set_processing/anomaly_indices/SPP_WR_load_anomaly_2021_30min/V_segments.jsonl"]'


#NORMAL_INDICES_FOR_SAMPLE='["./dataset_utils/ECG_datasets/indices_PV/slide_windows_2013_pv_sub_2npz/normal_200.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2013_pv_sub_3npz/normal_200.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2013_pv_sub_4npz/normal_200.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2015_pv_sub_0npz/normal_200.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2015_pv_sub_1npz/normal_200.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2021_pv_live_0npz/normal_200.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2022_pv_live_0npz/normal_200.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2023_pv_live_0npz/normal_200.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2024_pv_live_0npz/normal_200.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2025_pv_live_0npz/normal_200.jsonl"]'
NORMAL_INDICES_FOR_SAMPLE='["../data_set_processing/normal_indices_384/NYISO_CAPITL_load_anomaly_2024_30min/normal_384.jsonl","../data_set_processing/normal_indices_384/NYISO_CENTRL_load_anomaly_2024_30min/normal_384.jsonl","../data_set_processing/normal_indices_384/NYISO_DUNWOD_load_anomaly_2024_30min/normal_384.jsonl","../data_set_processing/normal_indices_384/NYISO_GENESE_load_anomaly_2024_30min/normal_384.jsonl","../data_set_processing/normal_indices_384/NYISO_HUD_VL_load_anomaly_2024_30min/normal_384.jsonl","../data_set_processing/normal_indices_384/NYISO_LONGIL_load_anomaly_2024_30min/normal_384.jsonl","../data_set_processing/normal_indices_384/NYISO_MHK_VL_load_anomaly_2024_30min/normal_384.jsonl","../data_set_processing/normal_indices_384/NYISO_MILLWD_load_anomaly_2024_30min/normal_384.jsonl","../data_set_processing/normal_indices_384/NYISO_NORTH_load_anomaly_2024_30min/normal_384.jsonl","../data_set_processing/normal_indices_384/NYISO_N_Y_C_load_anomaly_2024_30min/normal_384.jsonl","../data_set_processing/normal_indices_384/NYISO_WEST_load_anomaly_2024_30min/normal_384.jsonl","../data_set_processing/normal_indices_384/PJM_MIDATL_load_anomaly_2024_30min/normal_384.jsonl","../data_set_processing/normal_indices_384/PJM_RTO_load_anomaly_2024_30min/normal_384.jsonl","../data_set_processing/normal_indices_384/PJM_SOUTH_load_anomaly_2024_30min/normal_384.jsonl","../data_set_processing/normal_indices_384/PJM_WEST_load_anomaly_2024_30min/normal_384.jsonl","../data_set_processing/normal_indices_384/SPP_CSWS_load_anomaly_2021_30min/normal_384.jsonl","../data_set_processing/normal_indices_384/SPP_EDE_load_anomaly_2021_30min/normal_384.jsonl","../data_set_processing/normal_indices_384/SPP_GRDA_load_anomaly_2021_30min/normal_384.jsonl","../data_set_processing/normal_indices_384/SPP_INDN_load_anomaly_2021_30min/normal_384.jsonl","../data_set_processing/normal_indices_384/SPP_KACY_load_anomaly_2021_30min/normal_384.jsonl","../data_set_processing/normal_indices_384/SPP_KCPL_load_anomaly_2021_30min/normal_384.jsonl","../data_set_processing/normal_indices_384/SPP_LES_load_anomaly_2021_30min/normal_384.jsonl","../data_set_processing/normal_indices_384/SPP_MPS_load_anomaly_2021_30min/normal_384.jsonl","../data_set_processing/normal_indices_384/SPP_NPPD_load_anomaly_2021_30min/normal_384.jsonl","../data_set_processing/normal_indices_384/SPP_OKGE_load_anomaly_2021_30min/normal_384.jsonl","../data_set_processing/normal_indices_384/SPP_OPPD_load_anomaly_2021_30min/normal_384.jsonl","../data_set_processing/normal_indices_384/SPP_SECI_load_anomaly_2021_30min/normal_384.jsonl","../data_set_processing/normal_indices_384/SPP_SPRM_load_anomaly_2021_30min/normal_384.jsonl","../data_set_processing/normal_indices_384/SPP_SPS_load_anomaly_2021_30min/normal_384.jsonl","../data_set_processing/normal_indices_384/SPP_WAUE_load_anomaly_2021_30min/normal_384.jsonl","../data_set_processing/normal_indices_384/SPP_WFEC_load_anomaly_2021_30min/normal_384.jsonl","../data_set_processing/normal_indices_384/SPP_WR_load_anomaly_2021_30min/normal_384.jsonl"]'


EVENT_LABELS_PATHS='["./dataset_utils/ECG_datasets/indices_PV/slide_windows_2013_pv_sub_2npz/event_label.npy","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2013_pv_sub_3npz/event_label.npy","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2013_pv_sub_4npz/event_label.npy","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2015_pv_sub_0npz/event_label.npy","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2015_pv_sub_1npz/event_label.npy","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2021_pv_live_0npz/event_label.npy","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2022_pv_live_0npz/event_label.npy","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2023_pv_live_0npz/event_label.npy","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2024_pv_live_0npz/event_label.npy","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2025_pv_live_0npz/event_label.npy"]'

#VQVAE_TRAIN_INDICES_PATHS='["./dataset_utils/ECG_datasets/indices_PV/slide_windows_2013_pv_sub_2npz/mixed.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2013_pv_sub_3npz/mixed.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2013_pv_sub_4npz/mixed.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2015_pv_sub_0npz/mixed.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2015_pv_sub_1npz/mixed.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2021_pv_live_0npz/mixed.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2022_pv_live_0npz/mixed.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2023_pv_live_0npz/mixed.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2024_pv_live_0npz/mixed.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2025_pv_live_0npz/mixed.jsonl"]'
VQVAE_TRAIN_INDICES_PATHS='["../data_set_processing/mixed_indices/NYISO_CAPITL_load_anomaly_2024_30min/mixed.jsonl","../data_set_processing/mixed_indices/NYISO_CENTRL_load_anomaly_2024_30min/mixed.jsonl","../data_set_processing/mixed_indices/NYISO_DUNWOD_load_anomaly_2024_30min/mixed.jsonl","../data_set_processing/mixed_indices/NYISO_GENESE_load_anomaly_2024_30min/mixed.jsonl","../data_set_processing/mixed_indices/NYISO_HUD_VL_load_anomaly_2024_30min/mixed.jsonl","../data_set_processing/mixed_indices/NYISO_LONGIL_load_anomaly_2024_30min/mixed.jsonl","../data_set_processing/mixed_indices/NYISO_MHK_VL_load_anomaly_2024_30min/mixed.jsonl","../data_set_processing/mixed_indices/NYISO_MILLWD_load_anomaly_2024_30min/mixed.jsonl","../data_set_processing/mixed_indices/NYISO_NORTH_load_anomaly_2024_30min/mixed.jsonl","../data_set_processing/mixed_indices/NYISO_N_Y_C_load_anomaly_2024_30min/mixed.jsonl","../data_set_processing/mixed_indices/NYISO_WEST_load_anomaly_2024_30min/mixed.jsonl","../data_set_processing/mixed_indices/PJM_MIDATL_load_anomaly_2024_30min/mixed.jsonl","../data_set_processing/mixed_indices/PJM_RTO_load_anomaly_2024_30min/mixed.jsonl","../data_set_processing/mixed_indices/PJM_SOUTH_load_anomaly_2024_30min/mixed.jsonl","../data_set_processing/mixed_indices/PJM_WEST_load_anomaly_2024_30min/mixed.jsonl","../data_set_processing/mixed_indices/SPP_CSWS_load_anomaly_2021_30min/mixed.jsonl","../data_set_processing/mixed_indices/SPP_EDE_load_anomaly_2021_30min/mixed.jsonl","../data_set_processing/mixed_indices/SPP_GRDA_load_anomaly_2021_30min/mixed.jsonl","../data_set_processing/mixed_indices/SPP_INDN_load_anomaly_2021_30min/mixed.jsonl","../data_set_processing/mixed_indices/SPP_KACY_load_anomaly_2021_30min/mixed.jsonl","../data_set_processing/mixed_indices/SPP_KCPL_load_anomaly_2021_30min/mixed.jsonl","../data_set_processing/mixed_indices/SPP_LES_load_anomaly_2021_30min/mixed.jsonl","../data_set_processing/mixed_indices/SPP_MPS_load_anomaly_2021_30min/mixed.jsonl","../data_set_processing/mixed_indices/SPP_NPPD_load_anomaly_2021_30min/mixed.jsonl","../data_set_processing/mixed_indices/SPP_OKGE_load_anomaly_2021_30min/mixed.jsonl","../data_set_processing/mixed_indices/SPP_OPPD_load_anomaly_2021_30min/mixed.jsonl","../data_set_processing/mixed_indices/SPP_SECI_load_anomaly_2021_30min/mixed.jsonl","../data_set_processing/mixed_indices/SPP_SPRM_load_anomaly_2021_30min/mixed.jsonl","../data_set_processing/mixed_indices/SPP_SPS_load_anomaly_2021_30min/mixed.jsonl","../data_set_processing/mixed_indices/SPP_WAUE_load_anomaly_2021_30min/mixed.jsonl","../data_set_processing/mixed_indices/SPP_WFEC_load_anomaly_2021_30min/mixed.jsonl","../data_set_processing/mixed_indices/SPP_WR_load_anomaly_2021_30min/mixed.jsonl"]'


CODE_DIM=8
CODE_LEN=4

NUM_CODES=500



export CUDA_VISIBLE_DEVICES=0

VQVAE_CKPT="../formal_experiment/wenjia_power_0309/dsp_flow_mixed_K${NUM_CODES}/vqvae_save_path"
PRETRAIN_CKPT="../formal_experiment/wenjia_power_0309/dsp_flow_mixed_K${NUM_CODES}/no_context_pretrain_ckpt"
FINETUNE_CKPT="../formal_experiment/wenjia_power_0309/dsp_flow_mixed_K${NUM_CODES}/impute_finetune_ckpt_lr${LR}"

echo "Launching NUM_CODES=${NUM_CODES} on GPU ${GPU}"

python mini_runnable_vqvae.py \
  --wandb_project ${WANDB_PROJECT} \
  --wandb_run "VQVAE_mixed_-K${NUM_CODES}" \
  --max_seq_len ${MAX_LEN_ANOMALY} \
  --min_seq_len ${MIN_LEN_ANOMALY} \
  --data_paths ${DATA_PATHS} \
  --indices_paths ${VQVAE_TRAIN_INDICES_PATHS} \
  --data_type ${DATA_TYPE} \
  --gpu_id 0 \
  --save_dir ${VQVAE_CKPT} \
  --code_dim ${CODE_DIM} \
  --code_len ${CODE_LEN} \
  --num_codes ${NUM_CODES} \
  --one_channel ${ONE_CHANNEL} \
  --feat_size ${FEAT_SIZE}

python dsp_flow.py \
  --what_to_do "no_context_pretrain" \
  --num_codes ${NUM_CODES} \
  --seq_len ${LEN_WHOLE} \
  --data_type ${DATA_TYPE} \
  --feature_size ${FEAT_SIZE} \
  --one_channel ${ONE_CHANNEL} \
  --n_layer_enc 4 \
  --n_layer_dec 4 \
  --d_model 64 \
  --n_heads 4 \
  --raw_data_paths_train ${DATA_PATHS} \
  --raw_data_paths_test ${TEST_DATA_PATHS} \
  --event_labels_paths_train ${EVENT_LABELS_PATHS} \
  --indices_paths_train ${PRETRAIN_INDICES_PATHS} \
  --indices_paths_test "[]" \
  --indices_paths_anomaly_for_sample "[]" \
  --min_infill_length ${MIN_LEN_ANOMALY} \
  --max_infill_length ${MAX_LEN_ANOMALY} \
  --lr 1e-4 \
  --batch_size 64 \
  --max_epochs 100 \
  --grad_clip_norm 1.0 \
  --grad_accum_steps 1 \
  --early_stop "true" \
  --patience 50 \
  --wandb_project ${WANDB_PROJECT} \
  --wandb_run "no_context_pretrain_mixed_K${NUM_CODES}" \
  --ckpt_dir ${PRETRAIN_CKPT} \
  --pretrained_ckpt "none" \
  --vqvae_ckpt "${VQVAE_CKPT}/vqvae.pt" \
  --generated_path "none" \
  --gpu_id 0

python dsp_flow.py \
  --what_to_do "imputation_finetune" \
  --num_codes ${NUM_CODES} \
  --seq_len ${LEN_WHOLE} \
  --data_type ${DATA_TYPE} \
  --feature_size ${FEAT_SIZE} \
  --one_channel ${ONE_CHANNEL} \
  --n_layer_enc 4 \
  --n_layer_dec 4 \
  --d_model 64 \
  --n_heads 4 \
  --raw_data_paths_train ${DATA_PATHS} \
  --raw_data_paths_test ${TEST_DATA_PATHS} \
  --event_labels_paths_train ${EVENT_LABELS_PATHS} \
  --indices_paths_train ${FINETUNE_TRAIN_INDICES_PATHS} \
  --indices_paths_test ${FINETUNE_TEST_INDICES_PATHS} \
  --indices_paths_anomaly_for_sample "[]" \
  --min_infill_length ${MIN_LEN_ANOMALY} \
  --max_infill_length ${MAX_LEN_ANOMALY} \
  --lr ${LR} \
  --batch_size 64 \
  --max_epochs 500 \
  --grad_clip_norm 1.0 \
  --grad_accum_steps 1 \
  --early_stop "true" \
  --patience 500 \
  --wandb_project ${WANDB_PROJECT} \
  --wandb_run "impute_finetune_mixed_lr${LR}_K${NUM_CODES}" \
  --ckpt_dir ${FINETUNE_CKPT} \
  --pretrained_ckpt ${PRETRAIN_CKPT} \
  --vqvae_ckpt "${VQVAE_CKPT}/vqvae.pt" \
  --generated_path "none" \
  --gpu_id 0

  python dsp_flow.py \
    --what_to_do "posterior_impute_sample" \
    \
    --num_codes ${NUM_CODES} \
    --seq_len ${LEN_WHOLE} \
    --data_type ${DATA_TYPE} \
    --feature_size ${FEAT_SIZE} \
    --one_channel ${ONE_CHANNEL} \
    \
    --n_layer_enc 4 \
    --n_layer_dec 4 \
    --d_model 64 \
    --n_heads 4 \
    \
    --raw_data_paths_train ${DATA_PATHS} \
    --raw_data_paths_test ${TEST_DATA_PATHS} \
    --event_labels_paths_train ${EVENT_LABELS_PATHS} \
    --indices_paths_train ${NORMAL_INDICES_FOR_SAMPLE} \
    --indices_paths_test "[]" \
    --indices_paths_anomaly_for_sample ${ANOMALY_INDICES_FOR_SAMPLE} \
    --min_infill_length ${MIN_LEN_ANOMALY} \
    --max_infill_length ${MAX_LEN_ANOMALY} \
    \
    --lr 1e-4 \
    --batch_size 64 \
    --max_epochs 2000 \
    --grad_clip_norm 1.0 \
    --grad_accum_steps 1 \
    --early_stop "true" \
    --patience 50 \
    \
    --wandb_project "none" \
    --wandb_run "none" \
    \
    --ckpt_dir "${FINETUNE_CKPT}" \
    --pretrained_ckpt "none" \
    --vqvae_ckpt "${VQVAE_CKPT}/vqvae.pt" \
    \
    --generated_path "" \
    \
    --gpu_id 0

#  python dsp_flow.py \
#    --what_to_do "posterior_impute_sample_non_downstream" \
#    \
#    --num_codes ${NUM_CODES} \
#    --seq_len ${LEN_WHOLE} \
#    --data_type ${DATA_TYPE} \
#    --feature_size ${FEAT_SIZE} \
#    --one_channel ${ONE_CHANNEL} \
#    \
#    --n_layer_enc 4 \
#    --n_layer_dec 4 \
#    --d_model 64 \
#    --n_heads 4 \
#    \
#    --raw_data_paths_train ${DATA_PATHS} \
#    --raw_data_paths_test ${TEST_DATA_PATHS} \
#    --event_labels_paths_train ${EVENT_LABELS_PATHS} \
#    --indices_paths_train ${FINETUNE_TEST_INDICES_PATHS} \
#    --indices_paths_test "[]" \
#    --indices_paths_anomaly_for_sample ${ANOMALY_INDICES_FOR_SAMPLE} \
#    --min_infill_length ${MIN_LEN_ANOMALY} \
#    --max_infill_length ${MAX_LEN_ANOMALY} \
#    \
#    --lr 1e-4 \
#    --batch_size 64 \
#    --max_epochs 2000 \
#    --grad_clip_norm 1.0 \
#    --grad_accum_steps 1 \
#    --early_stop "true" \
#    --patience 50 \
#    \
#    --wandb_project "none" \
#    --wandb_run "none" \
#    \
#    --ckpt_dir ${FINETUNE_CKPT} \
#    --pretrained_ckpt "none" \
#    --vqvae_ckpt "${VQVAE_CKPT}/vqvae.pt" \
#    \
#    --generated_path "" \
#    \
#    --gpu_id 0

#  OUTDIR="../nn_eval/PV/dsp_flow_mixed_K${NUM_CODES}"
#  python run_nn_evaluate_new.py \
#    --seq_len ${LEN_WHOLE} \
#    --feature_size 1 \
#    --one_channel 1 \
#    --feat_window_size 100 \
#    --raw_data_paths ${DATA_PATHS} \
#    --indices_paths_test ${FINETUNE_TRAIN_INDICES_PATHS} \
#    --max_infill_length ${MAX_LEN_ANOMALY} \
#    --ckpt_dir "${FINETUNE_CKPT}" \
#    --out_dir "${OUTDIR}" \
#    --generated_path "${FINETUNE_CKPT}/principle_posterior_impute_samples.pth" \
#    --gpu_id 0


cd dsp_our_method
