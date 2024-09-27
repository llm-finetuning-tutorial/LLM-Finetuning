# LLM-Finetuning

## 빠른 시작

아래에서는 🤖 ModelScope와 🤗 Transformers를 사용하여 Qwen-Chat를 사용하는 간단한 예제를 제공합니다. 먼저 필요한 패키지를 설치하세요.

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
사용자가 쉽게 파인튜닝할 수 있도록 `finetune.py`를 제공합니다. 또한 편하게 파인튜닝을 시작할 수 있는 쉘 스크립트도 제공합니다. 이 스크립트는 [DeepSpeed](https://github.com/microsoft/DeepSpeed)와 [FSDP](https://engineering.fb.com/2021/07/15/open-source/fsdp/)를 사용한 훈련을 지원합니다. 제공된 쉘 스크립트는 DeepSpeed를 사용합니다 (참고: 이는 pydantic의 최신 버전과 충돌할 수 있으므로 `pydantic<2.0`을 사용해야 합니다)와 Peft. 다음과 같이 설치할 수 있습니다:
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
        "value": "안녕하세요"
      },
      {
        "from": "assistant",
        "value": "저는 언어 모델입니다."
      }
    ]
  }
]
```

데이터 준비 후, 제공된 쉘 스크립트를 사용하여 파인튜닝을 실행할 수 있습니다. 데이터 파일의 경로인 `$DATA`를 지정하세요.

파인튜닝 스크립트를 사용하면 다음을 수행할 수 있습니다:
- 전체 파라미터 파인튜닝
- LoRA
- Q-LoRA

### 전체 파라미터 파인튜닝
전체 파라미터 파인튜닝은 전체 훈련 과정에서 모든 파라미터를 업데이트해야 합니다. 훈련을 시작하려면 다음 스크립트를 실행하세요:

```bash
# 분산 훈련. 단일 GPU 훈련 스크립트는 제공하지 않습니다. GPU 메모리 부족으로 훈련이 중단될 수 있기 때문입니다.
bash finetune/finetune_ds.sh
```

쉘 스크립트에서 올바른 모델 이름 또는 경로, 데이터 경로, 출력 디렉토리를 지정하는 것을 잊지 마세요. 또 다른 주의할 점은 이 스크립트에서 DeepSpeed ZeRO 3를 사용한다는 것입니다. 변경하고 싶다면 `--deepspeed` 인수를 제거하거나 요구 사항에 따라 DeepSpeed 구성 json 파일을 변경하면 됩니다. 또한 이 스크립트는 혼합 정밀도 훈련을 지원하므로 `--bf16 True` 또는 `--fp16 True`를 사용할 수 있습니다. fp16을 사용할 때는 혼합 정밀도 훈련으로 인해 DeepSpeed를 사용해야 합니다. 경험적으로 기계가 bf16을 지원한다면 사전 훈련 및 정렬과 일관성을 유지하기 위해 bf16을 사용하는 것이 좋습니다. 따라서 기본적으로 bf16을 사용합니다.

### LoRA
마찬가지로 LoRA를 실행하려면 아래와 같이 다른 스크립트를 실행하세요. 시작하기 전에 `peft`를 설치했는지 확인하세요. 또한 모델, 데이터, 출력 경로를 지정해야 합니다. 사전 훈련된 모델에 대해 절대 경로를 사용하는 것이 좋습니다. LoRA는 어댑터만 저장하고 어댑터 구성 json 파일의 절대 경로는 로드할 사전 훈련된 모델을 찾는 데 사용되기 때문입니다. 또한 이 스크립트는 bf16과 fp16을 모두 지원합니다.

```bash
# 단일 GPU 훈련
bash finetune/finetune_lora_single_gpu.sh
# 분산 훈련
bash finetune/finetune_lora_ds.sh
```

전체 파라미터 파인튜닝과 비교하여 LoRA ([논문](https://arxiv.org/abs/2106.09685))는 어댑터 레이어의 파라미터만 업데이트하고 원래의 대규모 언어 모델 레이어는 동결된 상태로 유지합니다. 이를 통해 메모리 비용과 계산 비용을 크게 줄일 수 있습니다.

LoRA를 사용하여 Qwen-7B와 같은 기본 언어 모델을 파인튜닝하는 경우 Qwen-7B-Chat와 같은 채팅 모델 대신 스크립트가 자동으로 임베딩 및 출력 레이어를 훈련 가능한 파라미터로 전환한다는 점에 유의하세요. 이는 기본 언어 모델이 ChatML 형식으로 인한 특수 토큰에 대한 지식이 없기 때문입니다. 따라서 모델이 토큰을 이해하고 예측할 수 있도록 이러한 레이어를 업데이트해야 합니다. 다시 말해, LoRA에서 훈련에 특수 토큰을 도입한다면 코드 내에서 `modules_to_save`를 설정하여 레이어를 훈련 가능한 파라미터로 설정해야 합니다. 또한 이러한 파라미터를 훈련 가능하게 만들면 ZeRO 3를 사용할 수 없으므로 기본적으로 스크립트에서 ZeRO 2를 사용합니다. 새로운 훈련 가능한 파라미터가 없다면 DeepSpeed 구성 파일을 변경하여 ZeRO 3로 전환할 수 있습니다. 또한 이러한 훈련 가능한 파라미터의 유무에 따라 LoRA의 메모리 사용량에 상당한 차이가 있음을 발견했습니다. 따라서 메모리에 문제가 있다면 채팅 모델에 대해 LoRA 파인튜닝을 수행하는 것이 좋습니다. 자세한 정보는 아래 프로필을 확인하세요.

### Q-LoRA
여전히 메모리가 부족하다면 Q-LoRA ([논문](https://arxiv.org/abs/2305.14314))를 고려할 수 있습니다. 양자화된 대규모 언어 모델과 페이지드 어텐션과 같은 다른 기술을 사용하여 메모리 비용을 더욱 줄일 수 있습니다.

참고: 단일 GPU Q-LoRA 훈련을 실행하려면 `pip` 또는 `conda`를 통해 `mpi4py`를 설치해야 할 수 있습니다.

Q-LoRA를 실행하려면 다음 스크립트를 직접 실행하세요:

```bash
# 단일 GPU 훈련
bash finetune/finetune_qlora_single_gpu.sh
# 분산 훈련
bash finetune/finetune_qlora_ds.sh
```

Q-LoRA의 경우, Qwen-7B-Chat-Int4와 같은 제공된 양자화 모델을 로드하는 것이 좋습니다. bf16 모델을 사용해서는 **안 됩니다**. 전체 파라미터 파인튜닝 및 LoRA와 달리 Q-LoRA에서는 fp16만 지원됩니다. 단일 GPU 훈련의 경우, torch amp로 인한 오류 발생 사례로 인해 DeepSpeed를 사용해야 합니다.  

또한 Q-LoRA의 경우 LoRA의 특수 토큰 문제가 여전히 존재합니다. 그러나 ChatML 형식의 특수 토큰을 학습한 채팅 모델에 대해서만 Int4 모델을 제공하므로 레이어에 대해 걱정할 필요가 없습니다. Int4 모델의 레이어는 훈련 가능해서는 안 되며, 따라서 훈련에 특수 토큰을 도입하면 Q-LoRA가 작동하지 않을 수 있습니다.

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

> 참고: `peft>=0.8.0`인 경우, 토크나이저도 함께 로드하려고 시도합니다. 하지만 `trust_remote_code=True` 없이 초기화되어 `ValueError: Tokenizer class QWenTokenizer does not exist or is not currently imported.` 오류가 발생할 수 있습니다. 현재로서는 `peft<0.8.0`으로 다운그레이드하거나 토크나이저 파일을 다른 곳으로 옮겨 이 문제를 해결할 수 있습니다.

어댑터를 병합하고 파인튜닝된 모델을 독립 실행형 모델로 저장하려면 (LoRA에서만 가능하며, Q-LoRA의 파라미터는 병합할 수 없음) 다음 코드를 실행하세요:

```python
from peft import AutoPeftModelForCausalLM

model = AutoPeftModelForCausalLM.from_pretrained(
    path_to_adapter, # 출력 디렉토리 경로
    device_map="auto",
    trust_remote_code=True
).eval()

merged_model = model.merge_and_unload()
# max_shard_size와 safe_serialization은 필수가 아닙니다. 
# 각각 체크포인트 분할과 모델을 safetensors로 저장하는 데 사용됩니다.
merged_model.save_pretrained(new_model_directory, max_shard_size="2048MB", safe_serialization=True)
```

`new_model_directory` 디렉토리에는 병합된 모델 가중치와 모듈 파일이 포함됩니다. 저장된 파일에서 `*.cu`와 `*.cpp` 파일이 누락될 수 있으니 주의하세요. KV 캐시 기능을 사용하려면 이 파일들을 수동으로 복사해야 합니다. 또한, 이 단계에서는 토크나이저 파일이 새 디렉토리에 저장되지 않습니다. 토크나이저 파일을 복사하거나 다음 코드를 사용할 수 있습니다:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    path_to_adapter, # 출력 디렉토리 경로
    trust_remote_code=True
)

tokenizer.save_pretrained(new_model_directory)
```

참고: 다중 GPU 훈련의 경우, 사용 중인 기기에 맞는 분산 훈련용 하이퍼파라미터를 지정해야 합니다. 또한 데이터, 메모리 사용량, 훈련 속도를 고려하여 `--model_max_length` 인자로 최대 시퀀스 길이를 지정하는 것이 좋습니다.

### 파인튜닝된 모델 양자화하기

이 섹션은 전체 파라미터/LoRA 파인튜닝된 모델에 적용됩니다. (참고: Q-LoRA로 파인튜닝된 모델은 이미 양자화되어 있으므로 추가 양자화가 필요하지 않습니다.)
LoRA를 사용한 경우, 양자화 전에 위의 지침에 따라 모델을 병합해주세요.

파인튜닝된 모델의 양자화에는 [auto_gptq](https://github.com/PanQiWei/AutoGPTQ)를 사용하는 것이 좋습니다.

```bash
pip install auto-gptq optimum
```

참고: 현재 AutoGPTQ에는 [이 이슈](https://github.com/PanQiWei/AutoGPTQ/issues/370)에서 언급된 버그가 있습니다. [이 PR](https://github.com/PanQiWei/AutoGPTQ/pull/495)에서 해결 방법을 확인할 수 있으며, 이 브랜치를 가져와 소스에서 설치할 수 있습니다.

먼저, 보정 데이터를 준비하세요. 파인튜닝에 사용한 데이터를 재사용하거나 같은 형식의 다른 데이터를 사용할 수 있습니다.

그 다음, 아래 스크립트를 실행하세요:

```bash
python run_gptq.py \
    --model_name_or_path $YOUR_LORA_MODEL_PATH \
    --data_path $DATA \
    --out_path $OUTPUT_PATH \
    --bits 4 # int4의 경우 4, int8의 경우 8
```

이 단계는 GPU가 필요하며, 데이터 크기와 모델 크기에 따라 몇 시간이 소요될 수 있습니다.

그 다음, 모든 `*.py`, `*.cu`, `*.cpp` 파일과 `generation_config.json`을 출력 경로로 복사하세요. 그리고 해당하는 공식 양자화 모델에서 `config.json` 파일을 복사하여 덮어쓰는 것이 좋습니다
(예를 들어, `Qwen-7B-Chat`을 파인튜닝하고 `--bits 4`를 사용한 경우, [Qwen-7B-Chat-Int4](https://huggingface.co/Qwen/Qwen-7B-Chat-Int4/blob/main/config.json)에서 `config.json`을 찾을 수 있습니다).
또한 `gptq.safetensors`를 `model.safetensors`로 이름을 변경해야 합니다.

마지막으로, 공식 양자화 모델을 로드하는 것과 같은 방법으로 모델을 테스트해보세요. 예를 들면:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

tokenizer = AutoTokenizer.from_pretrained("/path/to/your/model", trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    "/path/to/your/model",
    device_map="auto",
    trust_remote_code=True
).eval()

response, history = model.chat(tokenizer, "안녕하세요", history=None)
print(response)
```