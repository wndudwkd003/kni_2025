import torch
from src.data.prompt_manager import PromptManager
from src.data.datasets.base_dataset import BaseDataset, check_limit_length

DEBUG = True

class SFTDataset(BaseDataset):

    def _build_user_content(self, sample_input: dict[str, str]):
        # 타입별 지시사항 가져오기
        type_instruction = PromptManager.get_type_instructions(self.config_manager.system.prompt_version)

        # 사용자 프롬프트 생성 앞 문장, 뒷 문장 포함
        user_content = f"{type_instruction} 앞 문장: {sample_input['front']}, 뒷 문장: {sample_input['back']}"
        return user_content.strip()

    def _build_answer_text(self, answer):
        # is_cot = self.config_manager.system.is_cot
        answer = answer + self.tokenizer.eos_token
        return answer


    def process_sample(self, sample):
        sample_id = sample["id"]
        inputs = sample["input"]
        outputs = sample.get("output", "")

        samples = []

        system_prompt = PromptManager.get_system_prompt(self.config_manager.system.prompt_version)
        user_content = self._build_user_content(inputs)

        source = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content}],
            add_generation_prompt=True,
            return_tensors="pt",
        )

        if isinstance(source, dict):
            source_ids = source["input_ids"][0].to(dtype=torch.long)
            # if DEBUG: print("source is dict")
        else:
            source_ids = source[0].to(dtype=torch.long)
            # if DEBUG: print("source is list")


        if self.is_train:
            answer_text = self._build_answer_text(outputs)
            target = self.tokenizer(
                answer_text,
                return_attention_mask=False,
                add_special_tokens=False,
                return_tensors="pt"
            )
            target["input_ids"] = target["input_ids"].type(torch.int64)

            input_ids = torch.concat((source_ids, target["input_ids"][0]))

            labels = torch.concat((
                torch.LongTensor([self.IGNORE_INDEX] * source_ids.shape[0]),
                target["input_ids"][0]
            ))

            # 패딩을 하지 않으므로 전 토큰 유효
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)

            samples.append({
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask
            })

            if DEBUG:
                print(f"Sample ID: {sample_id}")
                print(f"System prompt: {system_prompt}")
                print(f"User content: {user_content}")
                print(f"Outputs: {outputs}")
                print("" + "-" * 50)

        else:
            # 평가/추론: source만 사용
            attention_mask = torch.ones_like(source_ids, dtype=torch.long)
            samples.append({
                "id": sample_id,
                "input_ids": source_ids,
                "attention_mask": attention_mask
            })

        return samples

