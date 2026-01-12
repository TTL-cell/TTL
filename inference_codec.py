import time
import os

import torch
import torch.distributed as dist

import json
from arguments import get_args

from utils import print_args, initialize
from utils import save_rank

from pretrain.inferer import PretrainLMInferer, PretrainGenInferer


torch.set_num_threads(16)


# =============================================================================
# [Custom Class] SparkTTS용 Inferer (Text Masking 적용)
# =============================================================================
class SparkTTSLMInferer(PretrainLMInferer):
    """
    Spark-TTS 구조에 맞춰 오디오 부분의 Loss만 계산하도록 수정한 Inferer.
    """
    def infer_one_batch(self, model_batch, no_model_batch):
        # 1. 원본 Input ID 가져오기
        input_ids = model_batch['input_ids']
        
        # 2. Labels 생성 (기존에는 모델 내부에서 input_ids를 복사해서 쓰지만, 여기서 직접 제어)
        # 기본적으로 input_ids를 복사합니다.
        labels = input_ids.clone()
        
        # 3. Masking 로직: <|start_global_token|> 이전(Text 부분)은 전부 -100 처리
        # 토크나이저에서 Separator ID 찾기
        try:
            sep_token = "<|start_global_token|>"
            sep_id = self.tokenizer.convert_tokens_to_ids(sep_token)
            if sep_id is None: # 혹시 None이면 인코딩으로 찾기
                sep_id = self.tokenizer.encode(sep_token, add_special_tokens=False)[-1]
        except Exception as e:
            if dist.get_rank() == 0:
                print(f"[Warning] Could not find separator '{sep_token}'. Calculating full loss. Error: {e}")
            sep_id = -1

        # 배치 내 각 샘플에 대해 마스킹 적용
        if sep_id != -1:
            for i in range(len(input_ids)):
                # 해당 샘플에서 sep_token이 위치한 인덱스들 찾기
                sep_indices = (input_ids[i] == sep_id).nonzero(as_tuple=True)[0]
                
                if len(sep_indices) > 0:
                    # 첫 번째 등장하는 <|start_global_token|> 위치 찾기
                    start_idx = sep_indices[0]
                    # 처음부터 ~ start_idx 까지 -100으로 마스킹 (Loss 제외)
                    # 이렇게 하면 모델이 텍스트를 예측하는 부분은 점수에 반영되지 않습니다.
                    labels[i, :start_idx] = -100
                else:
                    # 오디오 토큰이 없는 데이터라면? (예외 처리: 전체 무시)
                    labels[i, :] = -100

        # 4. 수정된 Labels를 model_batch에 업데이트
        model_batch['labels'] = labels

        # 5. Loss 계산 (부모 클래스 메서드 호출)
        # mean=False로 하여 샘플별 Loss를 리턴받아야 함 (MiniPLM 요구사항)
        loss = self.compute_lm_loss(model_batch, no_model_batch, mean=False)
        
        if self.args.torch_compile is not None:
            loss = loss.clone()
            
        return loss


# =============================================================================
# Main Function
# =============================================================================

def grouped_infer(args, ds_config, device, inferer_cls, start, end, time_stamp):
    base_ckpt_path = args.model_path
    ckpt_paths = [os.path.join(
        base_ckpt_path, f"{s}") for s in range(start, end, 5000)]
    
    args.model_path = ckpt_paths[0]
    inferer = None
    base_save_path = args.save
    for i, ckpt_path in enumerate(ckpt_paths):
        args.model_path = ckpt_path
        args.save = os.path.join(base_save_path, os.path.basename(ckpt_path))
        os.makedirs(args.save, exist_ok=True)
        save_rank(time_stamp, os.path.join(args.save, "log.txt"))

        if i == 0:
            inferer = inferer_cls(args, ds_config, device)
        else:
            inferer.setup_model_and_optimizer(args, set_optim=False)
            
        inferer.inference()


def main():
    torch.backends.cudnn.enabled = False
    
    args = get_args()
    initialize(args)
    
    if dist.get_rank() == 0:
        print_args(args)
        with open(os.path.join(args.save, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=4)
    
    device = torch.cuda.current_device()
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    args.time_stamp = cur_time
    time_stamp = "\n\n" + "="*30 + f" EXP at {cur_time} " + "="*30
    save_rank(time_stamp, os.path.join(args.save, "log.txt"))
    
    with open(args.deepspeed_config, "r") as f:
        ds_config = json.load(f)

    ds_config["zero_optimization"]["stage"] = 0
    
    args.deepspeed_config = None
    
    # [수정] Inferer 클래스 선택 로직 변경
    if args.type == "pt_lm_infer":
        # 기존 PretrainLMInferer 대신 커스텀 SparkTTSLMInferer 사용
        inferer_cls = SparkTTSLMInferer 
    elif args.type == "pt_gen_infer":
        inferer_cls = PretrainGenInferer
    else:
        raise ValueError(f"Invalid type: {args.type}")     
    
    if args.grouped_infer:
        grouped_infer(args, ds_config, device, inferer_cls, args.ckpt_start, args.ckpt_end, time_stamp)
    else:
        inferer = inferer_cls(args, ds_config, device)
        inferer.inference()

    
if __name__ == "__main__":
    main()