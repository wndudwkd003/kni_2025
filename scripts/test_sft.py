import unsloth
from transformers import set_seed
from src.utils.seed_utils import set_all_seeds
from unsloth import FastModel
import os, json, argparse, hashlib, torch, re
from src.configs.config_manager import ConfigManager
from src.utils.data_utils import prepare_dataset
from src.utils.huggingface_utils import init_hub_env
from tqdm.auto import tqdm
from datetime import datetime


CURRENT_TEST_TYPE = "sft"

ALLOWED = {"순접", "역접", "양립"}

def get_valid_answer(answer: str) -> str:
    m = re.search(r"###\s*(.*?)\s*###", answer, flags=re.DOTALL)
    if m:
        token = m.group(1).strip()
        token = " ".join(token.split())  # 내부 공백 정규화
    else:
        token = answer.strip()

    # 라벨 정규화: 허용 라벨만 반환
    if token in ALLOWED:
        return token
    if "역접" in token:
        return "역접"
    if "순접" in token:
        return "순접"
    if "양립" in token:
        return "양립"
    return token  # 마지막 안전장치: 그래도 불일치면 원문 반환


def init_config_manager_for_test(save_dir: str = "configs") -> ConfigManager:
    # 테스트 환경에서는 저장된 설정을 불러옴
    cm = ConfigManager()
    config_dir = os.path.join(save_dir, "configs")
    cm.load_all_configs(config_dir=config_dir)

    adapter_dir = os.path.join(save_dir, "lora_adapter")
    test_result_dir = os.path.join(save_dir, "test_result")
    os.makedirs(test_result_dir, exist_ok=True)
    print(f"Test results will be saved to: {test_result_dir}")

    cm.update_config("system", {
        "save_dir": save_dir,
        "adapter_dir": adapter_dir,
        "test_result_dir": test_result_dir
    })

    cm.print_all_configs()
    return cm


def main(cm: ConfigManager):
    # 테스트 모드

    model_path = cm.system.adapter_dir if cm.model.full_finetune else cm.model.model_id
    model, tokenizer = FastModel.from_pretrained(
        model_name=model_path,
        max_seq_length=cm.model.max_seq_length,
        dtype=cm.model.dtype,
        load_in_4bit=cm.model.load_in_4bit,
        load_in_8bit=cm.model.load_in_8bit,
        full_finetuning=cm.model.full_finetune,
        trust_remote_code=True,
    )


    # 어댑터 로드
    model = FastModel.for_inference(model)
    # model.config.use_cache = False

    if not cm.model.full_finetune:
        model.load_adapter(cm.system.adapter_dir)

    test_f_p = os.path.join(cm.system.data_raw_dir, "test.json")
    with open(test_f_p, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    answer_results = {}

    # 테스트 데이터셋 로드
    test_dataset = prepare_dataset(
        config_manager=cm,
        tokenizer=tokenizer,
        task_type=CURRENT_TEST_TYPE,
        is_train=False
    )

    print(f"Test dataset size: {len(test_dataset)}")

    for sample in tqdm(test_dataset, desc="Testing", unit="sample"):
        sample_id = sample["id"]
        input_ids = sample["input_ids"].to(dtype=torch.long, device=model.device).unsqueeze(0)

        # Dataset에서 attention_mask를 빼버렸으니 여기서 직접 생성
        attention_mask = input_ids.ne(tokenizer.pad_token_id).to(dtype=torch.long, device=model.device)

        # 생성
        with torch.inference_mode():
            outputs = model.generate(
                input_ids,
                max_new_tokens=cm.model.max_new_tokens,
                do_sample=cm.model.do_sample,
                attention_mask=attention_mask,
                use_cache=cm.system.use_cache,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # 답변 추출
        gen_tokens = outputs[:, input_ids.size(1):]
        answer = tokenizer.decode(gen_tokens[0], skip_special_tokens=True).strip()

        original_answer = answer
        if cm.system.sentence_relationship:
            original_answer = answer
            answer = get_valid_answer(answer)

        print(f"Sample ID: {sample_id}, Original Answer: {original_answer}, Processed Answer: {answer}")

        answer_results[sample_id] = {
            "original_answer": original_answer,
            "answer": answer,
        }

    # 결과 정리
    for item in test_data:
        answer_result = answer_results[item["id"]]
        item["output"] = answer_result["answer"]
        item["original_output"] = answer_result["original_answer"]

    save_dir_hash = hashlib.md5(cm.system.save_dir.encode()).hexdigest()[:8]
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_filename = f"test_results_{save_dir_hash}_{timestamp}.json"
    output_path = os.path.join(cm.system.test_result_dir, output_filename)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {os.path.dirname(output_path)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test SFT Model")
    parser.add_argument("--save_dir", type=str, required=True, help="Must be set to save the trained model.")
    args = parser.parse_args()

    # 설정 관리자 초기화
    config_manager = init_config_manager_for_test(save_dir=args.save_dir)
    config_manager.update_config(CURRENT_TEST_TYPE, {"seed": config_manager.system.seed})
    init_hub_env(config_manager.system.hf_token)
    set_seed(config_manager.system.seed)
    # set_all_seeds(config_manager.system.seed, deterministic=config_manager.system.deterministic)
    main(config_manager)
