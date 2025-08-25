from enum import Enum

class PromptVersion(Enum):
    V0 = "nothing"
    V1 = "original"
    V2 = "sentence_relation"

class PromptManager:
    # 시스템 프롬프트
    SYSTEM_PROMPTS = {
        PromptVersion.V0: "",
        PromptVersion.V1: (
            "당신은 대화에서 문장 간 논리 관계 판별을 수행할 수 있는 어시스턴트 AI입니다. "
            "논리 관계 판별이란, 제시된 앞 문장과 뒤 문장의 내용을 바탕으로 두 문장 사이의 관계가 순접인지, 역접인지, 양립인지 판별합니다. "
        ),
        PromptVersion.V2: (
            "당신은 대화에서 문장 간 논리 관계 판별을 수행할 수 있는 어시스턴트 AI입니다. "
            "논리 관계 판별이란, 제시된 앞 문장과 뒤 문장의 내용을 바탕으로 두 문장 사이의 관계가 순접인지, 역접인지, 양립인지 판별합니다. "
        ),
    }

    # 질문 타입별 지시 정의
    TYPE_INSTRUCTIONS = {
        PromptVersion.V0: "",
        PromptVersion.V1: "지시사항: 문장 간 논리 관계를 판별하세요. 답변은 순접, 역접, 양립 중 하나로 출력해야합니다.",
        PromptVersion.V2: "지시사항: 문장 간 논리 관계를 판별하세요. 답변은 앞문장과 뒷문장을 자연스럽게 연결하는 접속사를 활용하여 정답으로 유도합니다. 최종 정답은 순접, 역접, 양립 중 하나로 출력해야합니다.",

    }

    @classmethod
    def get_system_prompt(cls, version: PromptVersion) -> str:
        """지정된 버전의 시스템 프롬프트 반환"""
        return cls.SYSTEM_PROMPTS[version]

    @classmethod
    def get_type_instructions(cls, version: PromptVersion) -> str:
        """지정된 버전의 타입별 instruction 반환"""
        return cls.TYPE_INSTRUCTIONS[version]

    # @classmethod
    # def get_instruction_for_type(cls, version: PromptVersion, question_type: str) -> str:
    #     """특정 버전과 질문 타입에 대한 instruction 반환"""
    #     type_instructions = cls.get_type_instructions(version)
    #     return type_instructions.get(question_type, "")
