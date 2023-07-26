import os
import datetime

N_EPOCH = 100
BATCH_SIZE = 128 # standard BATCH
# T = 31  #dynamic
D_LSTM = 64
D_BERT = 768
N_CLASS = 3 # 0to2 +3([PAD])
Learning_Rate = 1e-4
pretrained_name='bert-base-uncased'
D_Glove = 300
D_word2 = 512

DATASETS = "sim-R"
DATA_TRAIN_PATH = f"./Data/simulated-dialogue-master/{DATASETS}/train.json"
DATA_TEST_PATH = f"./Data/simulated-dialogue-master/{DATASETS}/test.json"
now_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
logprint_path = os.path.abspath(f"./log/log_print_{now_str}.csv")
print(logprint_path)

# now = datetime.datetime.now()
# with open(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),logprint_path)), mode = "a", encoding= "shift_jis") as logf:
#     print(f"\n{datetime.datetime.now()}", file= logf)
#     print("試しに日本語")
#     print(f"BATCH_SIZE: {BATCH_SIZE}", file= logf)
#     print(f"N_EPOCH: {N_EPOCH}", file= logf)
#     # print(f"T: {T}")
#     print(f"D_BERT: {D_BERT}", file= logf)
#     print(f"Learning Rate: {Learning_Rate}", file= logf)
#     print(f"DATASETS: {DATASETS}", file= logf)
#     # print(f"D_LSTM: {D_LSTM}")
#     # print(f"N_CLASS: {N_CLASS}")
