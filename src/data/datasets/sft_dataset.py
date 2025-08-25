import torch
from src.data.prompt_manager import PromptManager
from src.data.datasets.base_dataset import BaseDataset, check_limit_length

DEBUG = True

MARKER_TEXT = "###"            # 스팬을 표시하는 마커

class SFTDataset(BaseDataset):

    def _build_user_content(self, sample_input: dict[str, str]):
        type_instruction = PromptManager.get_type_instructions(self.config_manager.system.prompt_version)
        user_content = f"{type_instruction} 앞 문장: {sample_input['front']}, 뒷 문장: {sample_input['back']}"
        return user_content.strip()

    def _build_answer_text(self, answer):
        answer = answer + self.tokenizer.eos_token
        return answer

    def _find_marker_span_token_indices(self, token_ids: torch.Tensor) -> list[int]:
        # target(라벨) 토큰들에서 MARKER_TEXT("###")가 최소 2번 등장한다고 가정
        # 첫 마커의 시작부터 마지막 마커의 끝까지(포함)를 스팬으로 간주
        marker_ids = self.tokenizer(
            MARKER_TEXT, add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False
        )["input_ids"]

        ids = token_ids.tolist()
        n = len(ids)
        m = len(marker_ids)

        hit_starts = []
        i = 0
        while i + m <= n:
            j = 0
            ok = True
            while j < m:
                if ids[i + j] != marker_ids[j]:
                    ok = False
                    j = m
                else:
                    j += 1
            if ok:
                hit_starts.append(i)
                i += m
            else:
                i += 1

        if len(hit_starts) < 2:
            return []

        first_start = hit_starts[0]
        last_start = hit_starts[-1]
        last_end = last_start + m - 1
        return list(range(first_start, last_end + 1))

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
        else:
            source_ids = source[0].to(dtype=torch.long)

        if self.is_train:
            answer_text = self._build_answer_text(outputs)
            target = self.tokenizer(
                answer_text,
                return_attention_mask=False,
                add_special_tokens=False,
                return_tensors="pt"
            )
            target_ids = target["input_ids"][0].type(torch.int64)

            # ── (1) 원본 샘플: 전체 문장에 대해 손실 계산 ──
            input_ids = torch.cat((source_ids, target_ids), dim=0)
            labels = torch.cat((
                torch.full((source_ids.size(0),), fill_value=self.IGNORE_INDEX, dtype=torch.long),
                target_ids
            ), dim=0)

            samples.append({
                "input_ids": input_ids,
                "labels": labels,
            })

            # ── (2) 스팬 전용 샘플: ###…### 구간만 유효 라벨로 남겨 추가 ──
            if self.config_manager.system.sentence_relationship and self.config_manager.system.span_extra_weight > 0:
                span_idx = self._find_marker_span_token_indices(target_ids)
                if len(span_idx) > 0:
                    span_labels_only = torch.full_like(target_ids, fill_value=self.IGNORE_INDEX)
                    for t in span_idx:
                        span_labels_only[t] = target_ids[t]
                    span_labels_full = torch.cat((
                        torch.full((source_ids.size(0),), fill_value=self.IGNORE_INDEX, dtype=torch.long),
                        span_labels_only
                    ), dim=0)
                    for _ in range(int(self.config_manager.system.span_extra_weight)):
                        samples.append({
                            "input_ids": input_ids,
                            "labels": span_labels_full,
                        })

            if DEBUG:
                print(f"[SFTDataset] Sample ID: {sample_id}")
                print(f"[SFTDataset] Outputs: {outputs}")
                print("-" * 40)

        else:
            samples.append({
                "id": sample_id,
                "input_ids": source_ids,
            })

        return samples
