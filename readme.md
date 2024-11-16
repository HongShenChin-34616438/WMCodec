# WMCodec
PyTorch Implementation of [WMCodec: End-to-End Neural Speech Codec with Deep Watermarking for Authenticity Verification](https://arxiv.org/abs/2409.12121)

## Quick Started
### Dependencies
  `pip install -r requirement.txt`

### Default Preparation
Modify the parameter `--input_training_file` `--input_validation_file` `--checkpoint_path` in `train.py`

Modify the parameter `--input_wavs_dir` `--output_dir` `--checkpoint_file` in `inference-at.py`

Modify the config.json

### Watermark config
The watermark configuration is in the `watermark.py` file, defaulting to 4-digit base-16.

### Train
 `python train.py `

### Test
 `python inference-at.py `

### Acknowledgements
This implementation uses parts of the code from the following Github repos: [https://github.com/zjzser/TraceableSpeech](https://github.com/zjzser/TraceableSpeech)

### Citations
If you find this code useful in your research, please consider citing:
```
 @misc{zhou2024wmcodecendtoendneuralspeech,
      title={WMCodec: End-to-End Neural Speech Codec with Deep Watermarking for Authenticity Verification}, 
      author={Junzuo Zhou and Jiangyan Yi and Yong Ren and Jianhua Tao and Tao Wang and Chu Yuan Zhang},
      year={2024},
      eprint={2409.12121},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2409.12121}, 
}
```
