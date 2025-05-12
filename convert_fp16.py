from safetensors.torch import load_file, save_file
import torch

# 1. 기존 .safetensors 모델 로드
input_path = "<input_model_path>.safetensors"
output_path = "<output_model_path>.safetensors"

state_dict = load_file(input_path)

# 2. float32 → float16으로 변환
fp16_state_dict = {
    k: v.half() if torch.is_tensor(v) and v.dtype == torch.float32 else v
    for k, v in state_dict.items()
}

# 3. 저장
save_file(fp16_state_dict, output_path)

print(f"Saved: {output_path}")
