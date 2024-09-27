# LLM-Finetuning

## 빠른 시작

아래에서는 🤖 ModelScope와 🤗 Transformers를 사용하여 Qwen-Chat를 사용하는 간단한 예제를 제공합니다.

환경 설정 단계의 대부분을 건너뛰려면 미리 빌드된 도커 이미지를 사용할 수 있습니다. 자세한 내용은 ["미리 빌드된 도커 이미지 사용하기"](#-docker) 섹션을 참조하세요.

도커를 사용하지 않는 경우, 환경을 설정하고 필요한 패키지를 설치했는지 확인하세요. 위의 요구 사항을 충족하는지 확인한 후 종속 라이브러리를 설치하세요.

```bash
pip install -r requirements.txt
```

기기가 fp16 또는 bf16을 지원하는 경우, 높은 효율성과 낮은 메모리 사용량을 위해 [flash-attention](https://github.com/Dao-AILab/flash-attention) (**현재 flash attention 2를 지원합니다.**)을 설치하는 것이 좋습니다. (**flash-attention은 선택 사항이며 설치하지 않아도 프로젝트는 정상적으로 실행될 수 있습니다**)

```bash
git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention && pip install .
# 아래는 선택 사항입니다. 설치에 시간이 걸릴 수 있습니다.
# pip install csrc/layer_norm
# flash-attn 버전이 2.1.1보다 높은 경우 다음은 필요하지 않습니다.
# pip install csrc/rotary
```

이제 ModelScope 또는 Transformers로 시작할 수 있습니다.

## 파인튜닝

### 사용법
이제 사용자가 다운스트림 애플리케이션을 위해 사전 훈련된 모델을 간단하게 파인튜닝할 수 있도록 공식 훈련 스크립트인 `finetune.py`를 제공합니다. 또한 걱정 없이 파인튜닝을 시작할 수 있는 쉘 스크립트도 제공합니다. 이 스크립트는 [DeepSpeed](https://github.com/microsoft/DeepSpeed)와 [FSDP](https://engineering.fb.com/2021/07/15/open-source/fsdp/)를 사용한 훈련을 지원합니다. 제공된 쉘 스크립트는 DeepSpeed를 사용합니다 (참고: 이는 pydantic의 최신 버전과 충돌할 수 있으므로 `pydantic<2.0`을 사용해야 합니다)와 Peft. 다음과 같이 설치할 수 있습니다:
```bash
pip install "peft<0.8.0" deepspeed
```

훈련 데이터를 준비하려면 모든 샘플을 리스트에 넣고 json 파일로 저장해야 합니다. 각 샘플은 id와 대화를 위한 리스트로 구성된 딕셔너리입니다. 아래는 1개의 샘플이 있는 간단한 예제 리스트입니다:
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

데이터 준비 후, 제공된 쉘 스크립트를 사용하여 파인튜닝을 실행할 수 있습니다. 데이터 파일의 경로인 `$DATA`를 지정하는 것을 잊지 마세요.

파인튜닝 스크립트를 사용하면 다음을 수행할 수 있습니다:
- 전체 파라미터 파인튜닝
- LoRA
- Q-LoRA

전체 파라미터 파인튜닝은 전체 훈련 과정에서 모든 파라미터를 업데이트해야 합니다. 훈련을 시작하려면 다음 스크립트를 실행하세요:

```bash
# 분산 훈련. 단일 GPU 훈련 스크립트는 제공하지 않습니다. GPU 메모리 부족으로 훈련이 중단될 수 있기 때문입니다.
bash finetune/finetune_ds.sh
```

쉘 스크립트에서 올바른 모델 이름 또는 경로, 데이터 경로, 출력 디렉토리를 지정하는 것을 잊지 마세요. 또 다른 주의할 점은 이 스크립트에서 DeepSpeed ZeRO 3를 사용한다는 것입니다. 변경하고 싶다면 `--deepspeed` 인수를 제거하거나 요구 사항에 따라 DeepSpeed 구성 json 파일을 변경하면 됩니다. 또한 이 스크립트는 혼합 정밀도 훈련을 지원하므로 `--bf16 True` 또는 `--fp16 True`를 사용할 수 있습니다. fp16을 사용할 때는 혼합 정밀도 훈련으로 인해 DeepSpeed를 사용해야 합니다. 경험적으로 기계가 bf16을 지원한다면 사전 훈련 및 정렬과 일관성을 유지하기 위해 bf16을 사용하는 것이 좋습니다. 따라서 기본적으로 bf16을 사용합니다.

마찬가지로 LoRA를 실행하려면 아래와 같이 다른 스크립트를 실행하세요. 시작하기 전에 `peft`를 설치했는지 확인하세요. 또한 모델, 데이터, 출력 경로를 지정해야 합니다. 사전 훈련된 모델에 대해 절대 경로를 사용하는 것이 좋습니다. LoRA는 어댑터만 저장하고 어댑터 구성 json 파일의 절대 경로는 로드할 사전 훈련된 모델을 찾는 데 사용되기 때문입니다. 또한 이 스크립트는 bf16과 fp16을 모두 지원합니다.

```bash
# 단일 GPU 훈련
bash finetune/finetune_lora_single_gpu.sh
# 분산 훈련
bash finetune/finetune_lora_ds.sh
```

전체 파라미터 파인튜닝과 비교하여 LoRA ([논문](https://arxiv.org/abs/2106.09685))는 어댑터 레이어의 파라미터만 업데이트하고 원래의 대규모 언어 모델 레이어는 동결된 상태로 유지합니다. 이를 통해 메모리 비용과 계산 비용을 크게 줄일 수 있습니다.

LoRA를 사용하여 Qwen-7B와 같은 기본 언어 모델을 파인튜닝하는 경우 Qwen-7B-Chat와 같은 채팅 모델 대신 스크립트가 자동으로 임베딩 및 출력 레이어를 훈련 가능한 파라미터로 전환한다는 점에 유의하세요. 이는 기본 언어 모델이 ChatML 형식으로 인한 특수 토큰에 대한 지식이 없기 때문입니다. 따라서 모델이 토큰을 이해하고 예측할 수 있도록 이러한 레이어를 업데이트해야 합니다. 다시 말해, LoRA에서 훈련에 특수 토큰을 도입한다면 코드 내에서 `modules_to_save`를 설정하여 레이어를 훈련 가능한 파라미터로 설정해야 합니다. 또한 이러한 파라미터를 훈련 가능하게 만들면 ZeRO 3를 사용할 수 없으므로 기본적으로 스크립트에서 ZeRO 2를 사용합니다. 새로운 훈련 가능한 파라미터가 없다면 DeepSpeed 구성 파일을 변경하여 ZeRO 3로 전환할 수 있습니다. 또한 이러한 훈련 가능한 파라미터의 유무에 따라 LoRA의 메모리 사용량에 상당한 차이가 있음을 발견했습니다. 따라서 메모리에 문제가 있다면 채팅 모델에 대해 LoRA 파인튜닝을 수행하는 것이 좋습니다. 자세한 정보는 아래 프로필을 확인하세요.

여전히 메모리가 부족하다면 Q-LoRA ([논문](https://arxiv.org/abs/2305.14314))를 고려할 수 있습니다. 이는 양자화된 대규모 언어 모델과 페이지드 어텐션과 같은 다른 기술을 사용하여 메모리 비용을 더욱 줄일 수 있습니다.

참고: 단일 GPU Q-LoRA 훈련을 실행하려면 `pip` 또는 `conda`를 통해 `mpi4py`를 설치해야 할 수 있습니다.

Q-LoRA를 실행하려면 다음 스크립트를 직접 실행하세요:

```bash
# 단일 GPU 훈련
bash finetune/finetune_qlora_single_gpu.sh
# 분산 훈련
bash finetune/finetune_qlora_ds.sh
```

Q-LoRA의 경우, Qwen-7B-Chat-Int4와 같은 제공된 양자화 모델을 로드하는 것이 좋습니다. bf16 모델을 사용해서는 **안 됩니다**. 전체 파라미터 파인튜닝 및 LoRA와 달리 Q-LoRA에서는 fp16만 지원됩니다. 단일 GPU 훈련의 경우, torch amp로 인한 오류 관찰로 인해 혼합 정밀도 훈련을 위해 DeepSpeed를 사용해야 합니다. 또한 Q-LoRA의 경우 LoRA의 특수 토큰 문제가 여전히 존재합니다. 그러나 ChatML 형식의 특수 토큰을 학습한 채팅 모델에 대해서만 Int4 모델을 제공하므로 레이어에 대해 걱정할 필요가 없습니다. Int4 모델의 레이어는 훈련 가능해서는 안 되며, 따라서 훈련에 특수 토큰을 도입하면 Q-LoRA가 작동하지 않을 수 있습니다.

> 참고: Hugging Face의 내부 메커니즘으로 인해 저장된 체크포인트에서 특정 비 Python 파일(예: `*.cpp` 및 `*.cu`)이 
> 누락될 수 있습니다. 이러한 파일을 다른 파일이 포함된 디렉토리에 수동으로 복사해야 할 수 있습니다.

전체 파라미터 파인튜닝과 달리 LoRA와 Q-LoRA의 훈련은 어댑터 파라미터만 저장합니다. 훈련이 Qwen-7B에서 시작된다고 가정하면 아래와 같이 파인튜닝된 모델을 로드하여 추론할 수 있습니다:

```python
from peft import AutoPeftModelForCausalLM

model = AutoPeftModelForCausalLM.from_pretrained(
    path_to_adapter, # 출력 디렉토리 경로
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
