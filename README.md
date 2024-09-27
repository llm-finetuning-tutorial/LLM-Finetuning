# LLM-Finetuning

## Quickstart

Below, we provide simple examples to show how to use Qwen-Chat with 🤖 ModelScope and 🤗 Transformers.

You can use our pre-built docker images to skip most of the environment setup steps, see Section ["Using Pre-built Docker Images"](#-docker) for more details. 

If not using docker, please make sure you have setup the environment and installed the required packages. Make sure you meet the above requirements, and then install the dependent libraries.

```bash
pip install -r requirements.txt
```

If your device supports fp16 or bf16, we recommend installing [flash-attention](https://github.com/Dao-AILab/flash-attention) (**we support flash attention 2 now.**) for higher efficiency and lower memory usage. (**flash-attention is optional and the project can run normally without installing it**)

```bash
git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention && pip install .
# Below are optional. Installing them might be slow.
# pip install csrc/layer_norm
# If the version of flash-attn is higher than 2.1.1, the following is not needed.
# pip install csrc/rotary
```

Now you can start with ModelScope or Transformers.

## Finetuning

### Usage
Now we provide the official training script, `finetune.py`, for users to finetune the pretrained model for downstream applications in a simple fashion. Additionally, we provide shell scripts to launch finetuning with no worries. This script supports the training with [DeepSpeed](https://github.com/microsoft/DeepSpeed) and [FSDP](https://engineering.fb.com/2021/07/15/open-source/fsdp/). The shell scripts that we provide use DeepSpeed (Note: this may have conflicts with the latest version of pydantic and you should use make sure `pydantic<2.0`) and Peft. You can install them by:
```bash
pip install "peft<0.8.0" deepspeed
```

To prepare your training data, you need to put all the samples into a list and save it to a json file. Each sample is a dictionary consisting of an id and a list for conversation. Below is a simple example list with 1 sample:
```json
[
  {
    "id": "identity_0",
    "conversations": [
      {
        "from": "user",
        "value": "你好"
      },
      {
        "from": "assistant",
        "value": "我是一个语言模型，我叫通义千问。"
      }
    ]
  }
]
```

After data preparation, you can use the provided shell scripts to run finetuning. Remember to specify the path to the data file, `$DATA`.

The finetuning scripts allow you to perform:
- Full-parameter finetuning
- LoRA
- Q-LoRA

Full-parameter finetuning requires updating all parameters in the whole training process. To launch your training, run the following script:

```bash
# Distributed training. We do not provide single-GPU training script as the insufficient GPU memory will break down the training.
bash finetune/finetune_ds.sh
```

Remember to specify the correct model name or path, the data path, as well as the output directory in the shell scripts. Another thing to notice is that we use DeepSpeed ZeRO 3 in this script. If you want to make changes, just remove the argument `--deepspeed` or make changes in the DeepSpeed configuration json file based on your requirements. Additionally, this script supports mixed-precision training, and thus you can use `--bf16 True` or `--fp16 True`. Remember to use DeepSpeed when you use fp16 due to mixed precision training. Empirically we advise you to use bf16 to make your training consistent with our pretraining and alignment if your machine supports bf16, and thus we use it by default.

Similarly, to run LoRA, use another script to run as shown below. Before you start, make sure that you have installed `peft`. Also, you need to specify your paths to your model, data, and output. We advise you to use absolute path for your pretrained model. This is because LoRA only saves the adapter and the absolute path in the adapter configuration json file is used for finding out the pretrained model to load. Also, this script support both bf16 and fp16.

```bash
# Single GPU training
bash finetune/finetune_lora_single_gpu.sh
# Distributed training
bash finetune/finetune_lora_ds.sh
```

In comparison with full-parameter finetuning, LoRA ([paper](https://arxiv.org/abs/2106.09685)) only updates the parameters of adapter layers but keeps the original large language model layers frozen. This allows much fewer memory costs and thus fewer computation costs. 

Note that if you use LoRA to finetune the base language model, e.g., Qwen-7B, instead of chat models, e.g., Qwen-7B-Chat, the script automatically switches the embedding and output layer as trainable parameters. This is because the base language model has no knowledge of special tokens brought by ChatML format. Thus these layers should be updated for the model to understand and predict the tokens. Or in another word, if your training brings in special tokens in LoRA, you should set the layers to trainable parameters by setting `modules_to_save` inside the code. Also, if we have these parameters trainable, it is not available to use ZeRO 3, and this is why we use ZeRO 2 in the script by default. If you do not have new trainable parameters, you can switch to ZeRO 3 by changing the DeepSpeed configuration file. Additionally, we find that there is a significant gap between the memory footprint of LoRA with and without these trainable parameters. Therefore, if you have trouble with memory, we advise you to LoRA finetune the chat models. Check the profile below for more information. 

If you still suffer from insufficient memory, you can consider Q-LoRA ([paper](https://arxiv.org/abs/2305.14314)), which uses the quantized large language model and other techniques such as paged attention to allow even fewer memory costs. 

Note: to run single-GPU Q-LoRA training, you may need to install `mpi4py` through `pip` or `conda`.

To run Q-LoRA, directly run the following script:

```bash
# Single GPU training
bash finetune/finetune_qlora_single_gpu.sh
# Distributed training
bash finetune/finetune_qlora_ds.sh
```

For Q-LoRA, we advise you to load our provided quantized model, e.g., Qwen-7B-Chat-Int4. You **SHOULD NOT** use the bf16 models. Different from full-parameter finetuning and LoRA, only fp16 is supported for Q-LoRA. For single-GPU training, we have to use DeepSpeed for mixed-precision training due to our observation of errors caused by torch amp. Besides, for Q-LoRA, the troubles with the special tokens in LoRA still exist. However, as we only provide the Int4 models for chat models, which means the language model has learned the special tokens of ChatML format, you have no worry about the layers. Note that the layers of the Int4 model should not be trainable, and thus if you introduce special tokens in your training, Q-LoRA might not work.

> NOTE: Please be aware that due to the internal mechanisms of Hugging Face, certain non-Python files (e.g., `*.cpp` and `*.cu`) 
> may be missing from the saved checkpoint. You may need to manually copy them to the directory containing other files.

Different from full-parameter finetuning, the training of both LoRA and Q-LoRA only saves the adapter parameters. Suppose your training starts from Qwen-7B, you can load the finetuned model for inference as shown below:

```python
from peft import AutoPeftModelForCausalLM

model = AutoPeftModelForCausalLM.from_pretrained(
    path_to_adapter, # path to the output directory
    device_map="auto",
    trust_remote_code=True
).eval()
```

> NOTE: If `peft>=0.8.0`, it will try to load the tokenizer as well, however, initialized without `trust_remote_code=True`, leading to `ValueError: Tokenizer class QWenTokenizer does not exist or is not currently imported.` Currently, you could downgrade `peft<0.8.0` or move tokenizer files elsewhere to workaround this issue.

If you want to merge the adapters and save the finetuned model as a standalone model (you can only do this with LoRA, and you CANNOT merge the parameters from Q-LoRA), you can run the following codes:

```python
from peft import AutoPeftModelForCausalLM

model = AutoPeftModelForCausalLM.from_pretrained(
    path_to_adapter, # path to the output directory
    device_map="auto",
    trust_remote_code=True
).eval()

merged_model = model.merge_and_unload()
# max_shard_size and safe serialization are not necessary. 
# They respectively work for sharding checkpoint and save the model to safetensors
merged_model.save_pretrained(new_model_directory, max_shard_size="2048MB", safe_serialization=True)
```

The `new_model_directory` directory will contain the merged model weights and module files. Please note that `*.cu` and `*.cpp` files may be missing in the saved files. If you wish to use the KV cache functionality, please manually copy them. Besides, the tokenizer files are not saved in the new directory in this step. You can copy the tokenizer files or use the following code
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    path_to_adapter, # path to the output directory
    trust_remote_code=True
)

tokenizer.save_pretrained(new_model_directory)
```


Note: For multi-GPU training, you need to specify the proper hyperparameters for distributed training based on your machine. Besides, we advise you to specify your maximum sequence length with the argument `--model_max_length`, based on your consideration of data, memory footprint, and training speed.

### Quantize Fine-tuned Models

This section applies to full-parameter/LoRA fine-tuned models. (Note: You do not need to quantize the Q-LoRA fine-tuned model because it is already quantized.)
If you use LoRA, please follow the above instructions to merge your model before quantization. 

We recommend using [auto_gptq](https://github.com/PanQiWei/AutoGPTQ) to quantize the finetuned model. 

```bash
pip install auto-gptq optimum
```

Note: Currently AutoGPTQ has a bug referred in [this issue](https://github.com/PanQiWei/AutoGPTQ/issues/370). Here is a [workaround PR](https://github.com/PanQiWei/AutoGPTQ/pull/495), and you can pull this branch and install from the source.

First, prepare the calibration data. You can reuse the fine-tuning data, or use other data following the same format.

Second, run the following script:

```bash
python run_gptq.py \
    --model_name_or_path $YOUR_LORA_MODEL_PATH \
    --data_path $DATA \
    --out_path $OUTPUT_PATH \
    --bits 4 # 4 for int4; 8 for int8
```

This step requires GPUs and may costs a few hours according to your data size and model size.

Then, copy all `*.py`, `*.cu`, `*.cpp` files and `generation_config.json` to the output path. And we recommend you to overwrite `config.json` by copying the file from the coresponding official quantized model
(for example, if you are fine-tuning `Qwen-7B-Chat` and use `--bits 4`, you can find the `config.json` from [Qwen-7B-Chat-Int4](https://huggingface.co/Qwen/Qwen-7B-Chat-Int4/blob/main/config.json)).
You should also rename the ``gptq.safetensors`` into ``model.safetensors``.

Finally, test the model by the same method to load the official quantized model. For example,

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

tokenizer = AutoTokenizer.from_pretrained("/path/to/your/model", trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    "/path/to/your/model",
    device_map="auto",
    trust_remote_code=True
).eval()

response, history = model.chat(tokenizer, "你好", history=None)
print(response)
```
