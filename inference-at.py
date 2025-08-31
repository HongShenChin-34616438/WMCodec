from __future__ import absolute_import, division, print_function, unicode_literals
import glob
import os
import argparse
import json
import time
from time import perf_counter
import pdb
import torch
from scipy.io.wavfile import write
from env import AttrDict
from meldataset import mel_spectrogram, MAX_WAV_VALUE, load_wav
from models import Generator, Encoder, Quantizer
from watermark import Random_watermark, Watermark_Encoder, Watermark_Decoder, sign_loss, attack, clip

h = None
sample_num = 10 
bit_num = 4 

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def get_mel(x):
    return mel_spectrogram(x, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]

def count_common_elements(tensorA, tensorB):
    cnt = 0
    for i in range(tensorA.size(1)):
        if tensorA[0][i] == tensorB[0][i]:
            cnt += 1
    return cnt


def inference(a):
    generator = Generator(h).to(device)
    encoder = Encoder(h).to(device)
    quantizer_Audio = Quantizer(h, 'Audio').to(device)
    watermark_encoder = Watermark_Encoder(h).to(device)
    watermark_decoder = Watermark_Decoder(h).to(device)
    state_dict_g = load_checkpoint(a.checkpoint_file, device)
    generator.load_state_dict(state_dict_g['generator'])
    encoder.load_state_dict(state_dict_g['encoder'])
    quantizer_Audio.load_state_dict(state_dict_g['quantizer_Audio'])
    watermark_encoder.load_state_dict(state_dict_g['watermark_encoder'])
    watermark_decoder.load_state_dict(state_dict_g['watermark_decoder'])

    filelist = os.listdir(a.input_wavs_dir)
    print("filelist: ", len(filelist))

    os.makedirs(a.output_dir, exist_ok=True)
    
    generator.eval()
    generator.remove_weight_norm()
    encoder.eval()
    encoder.remove_weight_norm()
    watermark_encoder.eval()
    watermark_decoder.eval()

    N_result_dic = {
    "CLP": [0, 0, 0],
    "RSP-90": [0, 0, 0],
    "Noise-W35": [0, 0, 0],
    "SS-01": [0, 0, 0],
    "AS-90": [0, 0, 0],
    "EA-0301": [0, 0, 0],
    "LP5000": [0, 0, 0]
    }

    Y_result_dic = {
    "CLP": [0, 0, 0],
    "RSP-90": [0, 0, 0],
    "Noise-W35": [0, 0, 0],
    "SS-01": [0, 0, 0],
    "AS-90": [0, 0, 0],
    "EA-0301": [0, 0, 0],
    "LP5000": [0, 0, 0]
    }

    short_time_raw_discard = []
    short_time_clip_discard = []

    print("device : ", device)
    with torch.no_grad():
        for i, filename in enumerate(filelist):
            wav, sr = load_wav(os.path.join(a.input_wavs_dir, filename))
            wav_length = len(wav)/sr
            wav = wav / MAX_WAV_VALUE
            wav = torch.FloatTensor(wav).to(device)
            y = wav.unsqueeze(0).unsqueeze(1)

            if y.shape[2] <= 1.125 * sr: # skip the length shorter than 1.125s 
                print("id : ", i, "filename : ", filename, " length is ", y.shape[2])
                short_time_raw_discard.append((i, filename, y.shape[2]))
                continue

            for j in range(sample_num):

                sign = Random_watermark(1).to(device)
                sign_en = watermark_encoder(sign)
                sign_trait = sign_en
                
                en_y = encoder(y, sign_en)
                q, loss_q, c =  quantizer_Audio(en_y) 
                # q = torch.stack([code.reshape(q.size(0), -1) for code in c], -1) 
                # q = quantizer_Audio.embed(q, h.Audio['infer_need_layer'])
            
                y_g_hat = generator(q)

                if j == 0: 
                    audio = y_g_hat.squeeze()
                    audio = audio * MAX_WAV_VALUE
                    audio = audio.cpu().numpy().astype('int16')
                    output_file = os.path.join(a.output_dir, os.path.splitext(filename)[0] + '.wav')
                    write(output_file, h.sampling_rate, audio)
                
                y_g_hat, clip_flag = clip(y_g_hat)
                y_g_hat, Opera = attack(y_g_hat, [("CLP", 0.13), ("RSP-90", 0.15), ("Noise-W35", 0.14), ("SS-01", 0.15), ("AS-90", 0.15), ("EA-0301", 0.14), ("LP5000", 0.14)]) # 施加攻击
                # complete attack
                
                if y_g_hat.shape[2] <= 1.125 * sr: 
                    print("id : ", i , "edit_id : ", j, "filename : ", filename, "clip_flag", clip_flag , "now length is ", y_g_hat.shape[2])
                    short_time_clip_discard.append((i , j , filename, clip_flag, y_g_hat.shape[2], Opera))
                    continue 
                
                y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size,
                                            h.fmin, h.fmax_for_loss) # 1024, 80, 24000, 240,1024 # [32, 80, 50]
                sign_score, sign_g_hat = watermark_decoder(y_g_hat_mel)
                audiomark_loss = sign_loss(sign_score, sign)

                if clip_flag == "N":
                    N_result_dic[f"{Opera}"][0] += 1 
                    N_result_dic[f"{Opera}"][1] += count_common_elements(sign, sign_g_hat) 
                    N_result_dic[f"{Opera}"][2] += audiomark_loss 

                if clip_flag == "Y":
                    Y_result_dic[f"{Opera}"][0] += 1
                    Y_result_dic[f"{Opera}"][1] += count_common_elements(sign, sign_g_hat) 
                    Y_result_dic[f"{Opera}"][2] += audiomark_loss 
                
                print("audio_id: ", i, "sign_id: ", j, 'sign: ', sign, "clip: ", clip_flag ,"Opera: ", Opera ,'cross_entropy: ', audiomark_loss, 'predict_sign: ', sign_g_hat)


        print("===============================")
        print("short_time_raw_discard length", len(short_time_raw_discard)) 
        print("short_time_clip_discard length", len(short_time_clip_discard))  
        print("===============================")
        print("No CLIP")
        print("total stastic:")
        print("audiomark_loss:")
        for Opera in [ "CLP", "RSP-90", "Noise-W35", "SS-01", "AS-90", "EA-0301", "LP5000"]:
            if N_result_dic[f"{Opera}"][0] > 0:
              print("Opera", Opera, "iter", N_result_dic[f"{Opera}"][0] ,"value", N_result_dic[f"{Opera}"][2] / N_result_dic[f"{Opera}"][0] )
            else:
              print("Opera", Opera, "iter", N_result_dic[f"{Opera}"][0], "value", "N/A (no samples)")
        print("ACC:")
        for Opera in [ "CLP", "RSP-90", "Noise-W35", "SS-01", "AS-90", "EA-0301", "LP5000"]:
            if N_result_dic[f"{Opera}"][0] > 0:
              print("Opera", Opera, "iter", N_result_dic[f"{Opera}"][0] ,"value", N_result_dic[f"{Opera}"][1] / (N_result_dic[f"{Opera}"][0] * bit_num) )
            else:
              print("Opera", Opera, "iter", N_result_dic[f"{Opera}"][0] ,"value", "N/A (no samples)")


        print("===============================")
        print("Yes CLIP")
        print("total stastic:")
        print("audiomark_loss:")
        for Opera in [ "CLP", "RSP-90", "Noise-W35", "SS-01", "AS-90", "EA-0301", "LP5000"]:
            if Y_result_dic[f"{Opera}"][0] > 0:
              print("Opera", Opera, "iter", Y_result_dic[f"{Opera}"][0] ,"value", Y_result_dic[f"{Opera}"][2] / Y_result_dic[f"{Opera}"][0] )
            else:
              print("Opera", Opera, "iter", Y_result_dic[f"{Opera}"][0], "value", "N/A (no samples)")
        print("ACC:")
        for Opera in [ "CLP", "RSP-90", "Noise-W35", "SS-01", "AS-90", "EA-0301", "LP5000"]:
            if Y_result_dic[f"{Opera}"][0] > 0:
              print("Opera", Opera, "iter", Y_result_dic[f"{Opera}"][0] ,"value", Y_result_dic[f"{Opera}"][1] / (Y_result_dic[f"{Opera}"][0] * bit_num) )
            else:
              print("Opera", Opera, "iter", Y_result_dic[f"{Opera}"][0] ,"value", "N/A (no samples)" )


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_wavs_dir', default='')
    parser.add_argument('--output_dir', default='')
    parser.add_argument('--checkpoint_file', default='') 
    a = parser.parse_args()

    config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    '''
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
    '''
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    inference(a)


if __name__ == '__main__':
    main()

