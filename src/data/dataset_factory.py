from src.data.datasets.sft_dataset import SFTDataset
from src.data.datasets.dpo_dataset import DPODataset
from src.configs.config_manager import ConfigManager

class DatasetFactory:
    DATASET_TYPES = {
        "sft": SFTDataset,
        "dpo": DPODataset,
        # "rag": RAGDataset,
    }

    @classmethod  # 이 데코레이터 추가!
    def create_dataset(
        cls,
        dataset_type: str,
        fname: str,
        tokenizer,
        config_manager: ConfigManager,
        data_shuffle: bool = False,
        is_train: bool = True
    ):
        """데이터셋 생성"""
        dataset_class = cls.DATASET_TYPES.get(dataset_type.lower())
        if not dataset_class:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

        return dataset_class(
            fname=fname,
            tokenizer=tokenizer,
            config_manager=config_manager,
            data_shuffle=data_shuffle,
            is_train=is_train,
            task_type=dataset_type
        )

