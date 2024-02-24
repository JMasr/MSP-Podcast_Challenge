# -*- coding: UTF-8 -*-
# Local modules
import os
import sys
import json
import argparse
import multiprocessing

# PyTorch Modules
import torch
from torch import nn
from torch.utils.data import DataLoader
# 3rd-Party Modules
from tqdm import tqdm
from transformers import AutoModel

from process_labels_for_categorical import process_labels_for_categorical

# Self-Written Modules
sys.path.append(os.getcwd())
import net
import utils

# Get number of cores for multi-threading
num_cores = multiprocessing.cpu_count() - 1

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--ssl_type", type=str, default="wav2vec2-xl")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--accumulation_steps", type=int, default=4)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--model_path", type=str, default="./temp")
parser.add_argument("--head_dim", type=int, default=32)
parser.add_argument("--num_layers", type=int, default=4)
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--nj", type=int, default=num_cores)
parser.add_argument("--pooling_type", type=str, default="MeanPooling")
args = parser.parse_args()

# Get the total number of available GPUs
num_gpus = torch.cuda.device_count()

if num_gpus == 0:
    print("No GPUs available. Make sure CUDA is properly installed.")
    device_ids = [torch.device("cpu")]
else:
    print("Number of available GPUs:", num_gpus)

    # Create a list of device IDs
    device_ids = list(range(num_gpus))
    print("Device IDs:", device_ids)

utils.set_deterministic(args.seed)
SSL_TYPE = utils.get_ssl_type(args.ssl_type)
assert SSL_TYPE != None, print("Invalid SSL type!")
BATCH_SIZE = args.batch_size
ACCUMULATION_STEP = args.accumulation_steps
assert (ACCUMULATION_STEP > 0) and (BATCH_SIZE % ACCUMULATION_STEP == 0)
EPOCHS = args.epochs
LR = args.lr
MODEL_PATH = args.model_path
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH, exist_ok=True)

config_path = "config.json"
with open(config_path, "r") as f:
    config = json.load(f)
audio_path = config["wav_dir"]
label_path = config["label_path"]

# Load the CSV file
df = process_labels_for_categorical(label_path)
label_path = label_path.replace('.csv', '_processed.csv')

# Filter out only 'Train' samples
train_df = df[df['Split_Set'] == 'Train']

# Classes (emotions)
classes = ['Angry', 'Sad', 'Happy', 'Surprise', 'Fear', 'Disgust', 'Contempt', 'Neutral']

# Calculate class frequencies
class_frequencies = train_df[classes].sum().to_dict()

# Total number of samples
total_samples = len(train_df)

# Calculate class weights
class_weights = {cls: total_samples / (len(classes) * freq) if freq != 0 else 0 for cls, freq in
                 class_frequencies.items()}

print(class_weights)

# Convert to list in the order of classes
weights_list = [class_weights[cls] for cls in classes]

# Convert to PyTorch tensor
class_weights_tensor = torch.tensor(weights_list, device='cuda', dtype=torch.float)

# Print or return the tensor
print(class_weights_tensor)

total_dataset = dict()
total_dataloader = dict()
for dtype in ["train", "dev"]:
    cur_utts, cur_labs = utils.load_cat_emo_label(label_path, dtype)
    cur_wavs = utils.load_audio(audio_path, cur_utts, args.nj)
    if dtype == "train":
        cur_wav_set = utils.WavSet(cur_wavs)
        cur_wav_set.save_norm_stat(MODEL_PATH + "/train_norm_stat.pkl")
    else:
        if dtype == "dev":
            wav_mean = total_dataset["train"].datasets[0].wav_mean
            wav_std = total_dataset["train"].datasets[0].wav_std
        elif dtype == "test":
            wav_mean, wav_std = utils.load_norm_stat(MODEL_PATH + "/train_norm_stat.pkl")
        cur_wav_set = utils.WavSet(cur_wavs, wav_mean=wav_mean, wav_std=wav_std)
    ########################################################
    cur_bs = BATCH_SIZE // ACCUMULATION_STEP if dtype == "train" else 1
    is_shuffle = True if dtype == "train" else False
    ########################################################
    cur_emo_set = utils.CAT_EmoSet(cur_labs)
    total_dataset[dtype] = utils.CombinedSet([cur_wav_set, cur_emo_set, cur_utts])
    total_dataloader[dtype] = DataLoader(
        total_dataset[dtype], batch_size=cur_bs, shuffle=is_shuffle,
        pin_memory=True, num_workers=4,
        collate_fn=utils.collate_fn_wav_lab_mask
    )

print("Loading pre-trained ", SSL_TYPE, " model...")

ssl_model = AutoModel.from_pretrained(SSL_TYPE)
ssl_model.freeze_feature_encoder()
ssl_model.eval()
ssl_model.cuda()

########## Implement pooling method ##########
feat_dim = ssl_model.config.hidden_size

pool_net = getattr(net, args.pooling_type)
attention_pool_type_list = ["AttentiveStatisticsPooling"]
if args.pooling_type in attention_pool_type_list:
    is_attentive_pooling = True
    pool_model = pool_net(feat_dim)
else:
    is_attentive_pooling = False
    pool_model = pool_net()
print(pool_model)
pool_model.cuda()

concat_pool_type_list = ["AttentiveStatisticsPooling"]
dh_input_dim = feat_dim * 2 \
    if args.pooling_type in concat_pool_type_list \
    else feat_dim

ser_model = net.EmotionRegression(dh_input_dim, args.head_dim, args.num_layers, 8, dropout=args.dropout)
##############################################
ser_model.eval()
ser_model.cuda()

ssl_opt = torch.optim.AdamW(ssl_model.parameters(), LR)
ser_opt = torch.optim.AdamW(ser_model.parameters(), LR)

# scaler = GradScaler()
ssl_opt.zero_grad(set_to_none=True)
ser_opt.zero_grad(set_to_none=True)

if is_attentive_pooling:
    pool_opt = torch.optim.AdamW(pool_model.parameters(), LR)
    pool_opt.zero_grad(set_to_none=True)

lm = utils.LogManager()
lm.alloc_stat_type_list(["train_loss"])
lm.alloc_stat_type_list(["dev_loss"])

min_epoch = 0
min_loss = 1e10

ssl_model = nn.DataParallel(ssl_model, device_ids=device_ids)
ssl_model.to(device_ids[0])

ser_model = nn.DataParallel(ser_model, device_ids=device_ids)
ser_model.to(device_ids[0])

pool_model = nn.DataParallel(pool_model, device_ids=device_ids)
pool_model.to(device_ids[0])

for epoch in range(EPOCHS):
    print("Epoch: ", epoch)
    lm.init_stat()
    ssl_model.train()
    pool_model.train()
    ser_model.train()
    batch_cnt = 0

    for xy_pair in tqdm(total_dataloader["train"]):
        x = xy_pair[0]
        x = x.cuda(non_blocking=True).float()
        y = xy_pair[1]
        y = y.max(dim=1)[1]
        y = y.cuda(non_blocking=True).long()
        mask = xy_pair[2]
        mask = mask.cuda(non_blocking=True).float()

        ssl = ssl_model(x, attention_mask=mask).last_hidden_state  # (B, T, 1024)
        ssl = pool_model(ssl, mask)

        emo_pred = ser_model(ssl)

        loss = utils.CE_weight_category(emo_pred, y, class_weights_tensor)

        total_loss = loss / ACCUMULATION_STEP
        total_loss.backward()
        if (batch_cnt + 1) % ACCUMULATION_STEP == 0 or (batch_cnt + 1) == len(total_dataloader["train"]):

            ssl_opt.step()

            ser_opt.step()

            if is_attentive_pooling:
                pool_opt.step()

            ssl_opt.zero_grad(set_to_none=True)
            ser_opt.zero_grad(set_to_none=True)
            if is_attentive_pooling:
                pool_opt.zero_grad(set_to_none=True)
        batch_cnt += 1

        # Logging
        lm.add_torch_stat("train_loss", loss)

    ssl_model.eval()
    pool_model.eval()
    ser_model.eval()
    total_pred = []
    total_y = []
    for xy_pair in tqdm(total_dataloader["dev"]):
        x = xy_pair[0]
        x = x.cuda(non_blocking=True).float()
        y = xy_pair[1]
        y = y.max(dim=1)[1]
        y = y.cuda(non_blocking=True).long()
        mask = xy_pair[2]
        mask = mask.cuda(non_blocking=True).float()

        with torch.no_grad():
            ssl = ssl_model(x, attention_mask=mask).last_hidden_state
            ssl = pool_model(ssl, mask)
            emo_pred = ser_model(ssl)

            total_pred.append(emo_pred)
            total_y.append(y)

    # CCC calculation
    total_pred = torch.cat(total_pred, 0)
    total_y = torch.cat(total_y, 0)
    loss = utils.CE_weight_category(emo_pred, y, class_weights_tensor)
    # Logging
    lm.add_torch_stat("dev_loss", loss)

    # Save model
    lm.print_stat()

    dev_loss = lm.get_stat("dev_loss")
    if min_loss > dev_loss:
        min_epoch = epoch
        min_loss = dev_loss

        print("Save", min_epoch)
        print("Loss", min_loss)
        save_model_list = ["ser", "ssl", "pool"]

        torch.save(ser_model.state_dict(),
                   os.path.join(MODEL_PATH, "final_ser.pt"))

        torch.save(ssl_model.state_dict(),
                   os.path.join(MODEL_PATH, "final_ssl.pt"))

        torch.save(pool_model.state_dict(),
                   os.path.join(MODEL_PATH, "final_pool.pt"))
