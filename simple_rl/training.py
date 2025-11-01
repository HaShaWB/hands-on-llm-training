"""
GRPO 학습 스크립트 (DeepSeek-R1 Zero 방식)
- 별도의 reward model 없이 규칙 기반 보상 함수 사용
- 객관식 문제에 대한 추론 능력 학습
"""

import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer


# ========================================
# 1. 데이터 로드 및 전처리
# ========================================

def load_training_data(json_file_path: str) -> Dataset:
    """
    JSON Lines 파일 또는 JSON 배열 파일을 로드하여 Dataset으로 변환

    Args:
        json_file_path: JSON 파일 경로

    Returns:
        Dataset: HuggingFace Dataset 객체
    """
    data_list = []

    # JSON Lines 형식 시도
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data_list.append(json.loads(line))
    except json.JSONDecodeError:
        # JSON 배열 형식으로 재시도
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data_list = json.load(f)

    # GRPO는 'prompt' 컬럼이 필요하므로 'query'를 'prompt'로 변경
    for item in data_list:
        item['prompt'] = item.pop('query')

    return Dataset.from_list(data_list)


def prepare_dataset_from_list(data_list: List[Dict[str, Any]]) -> Dataset:
    """
    Python 리스트에서 직접 Dataset 생성

    Args:
        data_list: 딕셔너리 리스트

    Returns:
        Dataset: HuggingFace Dataset 객체
    """
    # 'query'를 'prompt'로 변경
    processed_data = []
    for item in data_list:
        processed_item = item.copy()
        processed_item['prompt'] = processed_item.pop('query')
        processed_data.append(processed_item)

    return Dataset.from_list(processed_data)


# ========================================
# 2. 보상 함수 정의 (DeepSeek-R1 Zero 스타일)
# ========================================

def extract_final_answer(completion: str) -> str:
    """
    생성된 텍스트에서 최종 답안 추출

    Args:
        completion: 모델이 생성한 전체 텍스트

    Returns:
        str: 추출된 답안 (1-5 사이의 숫자)
    """
    # 일반적인 답안 패턴 매칭
    patterns = [
        r'(?:정답은|답은|답:|따라서)\s*([1-5])',
        r'(?:선택지|보기)\s*([1-5])',
        r'^([1-5])(?:\.|번)',
        r'([1-5])\s*$',
    ]

    for pattern in patterns:
        match = re.search(pattern, completion, re.MULTILINE)
        if match:
            return match.group(1)

    # 마지막 숫자 추출 시도
    numbers = re.findall(r'[1-5]', completion)
    if numbers:
        return numbers[-1]

    return ""


def accuracy_reward(completions: List[str], prompts: List[str], **kwargs) -> List[float]:
    """
    정답 정확도 기반 보상 함수

    Args:
        completions: 모델이 생성한 답변 리스트
        prompts: 원본 프롬프트 리스트 (사용하지 않음)
        **kwargs: 추가 정보 (answer 컬럼 포함)

    Returns:
        List[float]: 각 completion의 보상 값 (0.0 또는 1.0)
    """
    # kwargs에서 정답 가져오기
    answers = kwargs.get('answer', [])
    if not answers:
        return [0.0] * len(completions)

    rewards = []
    for completion, correct_answer in zip(completions, answers):
        predicted_answer = extract_final_answer(completion)

        # 정답 일치 여부로 보상
        if predicted_answer == str(correct_answer):
            rewards.append(1.0)
        else:
            rewards.append(0.0)

    return rewards


def format_reward(completions: List[str], **kwargs) -> List[float]:
    """
    형식 품질 기반 보상 함수
    - 추론 과정이 포함되어 있는지 평가
    - DeepSeek-R1 스타일의 사고 과정 장려

    Args:
        completions: 모델이 생성한 답변 리스트
        **kwargs: 추가 정보

    Returns:
        List[float]: 각 completion의 보상 값 (0.0 ~ 1.0)
    """
    rewards = []

    for completion in completions:
        score = 0.0

        # 1. 최소 길이 체크 (너무 짧은 답변 페널티)
        if len(completion) > 50:
            score += 0.2

        # 2. 추론 키워드 포함 여부
        reasoning_keywords = ['따라서', '그러므로', '왜냐하면', '때문에',
                              '먼저', '다음으로', '마지막으로',
                              '분석', '판단', '고려']
        if any(keyword in completion for keyword in reasoning_keywords):
            score += 0.3

        # 3. 문장 구조 품질 (문장 부호 사용)
        if '.' in completion or '?' in completion or '!' in completion:
            score += 0.2

        # 4. 답변 형식 포함 여부
        if re.search(r'[1-5]', completion):
            score += 0.3

        rewards.append(score)

    return rewards


def combined_reward(completions: List[str], prompts: List[str] = None, **kwargs) -> List[float]:
    """
    정확도와 형식을 결합한 보상 함수

    Args:
        completions: 모델이 생성한 답변 리스트
        prompts: 원본 프롬프트 리스트
        **kwargs: 추가 정보

    Returns:
        List[float]: 각 completion의 최종 보상 값
    """
    # 정확도 보상 (가중치: 0.7)
    acc_rewards = accuracy_reward(completions, prompts, **kwargs)

    # 형식 보상 (가중치: 0.3)
    fmt_rewards = format_reward(completions, **kwargs)

    # 가중 합산
    final_rewards = [
        0.7 * acc + 0.3 * fmt
        for acc, fmt in zip(acc_rewards, fmt_rewards)
    ]

    return final_rewards


# ========================================
# 3. 모델 및 학습 설정
# ========================================

def setup_model_with_lora(
        model_name: str,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05
):
    """
    LoRA를 적용한 모델 설정

    Args:
        model_name: 모델 이름 (예: "Qwen/Qwen2.5-0.5B-Instruct")
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout

    Returns:
        tuple: (model, tokenizer)
    """
    # 모델 로드
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    # padding token 설정
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA 설정
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # LoRA 적용
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


# ========================================
# 4. 메인 학습 함수
# ========================================

def train_grpo(
        data_path: str = None,
        data_list: List[Dict] = None,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        output_dir: str = "./grpo_output",
        num_epochs: int = 1,
        batch_size: int = 2,
        learning_rate: float = 5e-6,
        num_generations: int = 4,
        max_prompt_length: int = 512,
        max_completion_length: int = 256,
        use_lora: bool = True,
):
    """
    GRPO 학습 실행 함수

    Args:
        data_path: JSON 파일 경로
        data_list: 직접 제공하는 데이터 리스트
        model_name: 사용할 베이스 모델
        output_dir: 체크포인트 저장 디렉토리
        num_epochs: 학습 에포크 수
        batch_size: 배치 크기
        learning_rate: 학습률
        num_generations: 각 프롬프트당 생성 개수
        max_prompt_length: 최대 프롬프트 길이
        max_completion_length: 최대 생성 길이
        use_lora: LoRA 사용 여부
    """
    # 1. 데이터 로드
    if data_path:
        dataset = load_training_data(data_path)
    elif data_list:
        dataset = prepare_dataset_from_list(data_list)
    else:
        raise ValueError("data_path 또는 data_list 중 하나는 반드시 제공해야 합니다.")

    print(f"데이터셋 크기: {len(dataset)}")
    print(f"데이터 샘플:\n{dataset[0]}\n")

    # 2. 모델 설정
    if use_lora:
        model, tokenizer = setup_model_with_lora(model_name)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    # 3. GRPO 설정
    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        num_train_epochs=num_epochs,

        # GRPO 특화 파라미터
        num_generations=num_generations,  # 각 프롬프트당 생성 개수
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        temperature=0.7,  # 생성 다양성
        beta=0.04,  # KL divergence penalty

        # 최적화 설정
        optim="adamw_8bit",
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        max_grad_norm=0.5,

        # 로깅 및 저장
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        report_to="tensorboard",

        # 정밀도
        bf16=True,

        # 메모리 최적화
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        # 기타
        remove_unused_columns=False,  # answer 컬럼 유지
    )

    # 4. Trainer 초기화
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=combined_reward,  # 또는 [accuracy_reward, format_reward]
        tokenizer=tokenizer,
    )

    # 5. 학습 시작
    print("\n" + "="*50)
    print("GRPO 학습 시작")
    print("="*50 + "\n")

    trainer.train()

    # 6. 모델 저장
    print("\n학습 완료! 최종 모델 저장 중...")
    trainer.save_model(os.path.join(output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final_model"))

    print(f"모델이 {output_dir}/final_model 에 저장되었습니다.")


# ========================================
# 5. 사용 예시
# ========================================

if __name__ == "__main__":
    # 예시 데이터 (실제 사용 시 파일에서 로드)
    example_data = [
        {
            "query": "다음을 읽고 물음에 답하시오\n<지문>\n미술사를 다루고 있는 좋은 책이 많지만 학술적인 지식이 부족하면 이해하기 어려운 경우가 많다고 한다. 이런 점에서 미술에 대해 막 알아 가기 시작한 나와 같은 독자도 이해할 수 있다고 알려진, 곰브리치의 『서양 미술사』를 택해 서양 미술의 흐름을 살펴본 것은 좋은 결정이었다. 이 책을 통해 저자는 미술사를 어떻게 이해할 것인가를 설명한다. 저자는 서론에서 '미술이라는 것은 사실상 존재하지 않는다. 다만 미술가들이 있을 뿐이다.'라고 밝히며, 미술가와 미술 작품에 주목하여 미술사를 이해하려는 자신의 관점을 설명한다. 저자는 27장에서도 해당 구절을 들어 자신의 관점을 다시 설명하고 있었기 때문에, 27장의 내용을 서론의 내용과 비교하여 읽으면서 저자의 관점을 더 잘 이해할 수 있었다. 책의 제목을 처음 접했을 때는, 이 책이 유럽만을 대상으로 삼고 있을 거라고 생각했다. 하지만 책의 본문을 읽기 전에 목차를 살펴보니, 총 28장으로 구성된 이 책이 유럽 외의 지역도 포함하고 있음을 알 수 있었다. 1~7장에서는 아메리카, 이집트, 중국 등의 미술도 설명하고 있었고, 8~28장에서는 6세기 이후 유럽 미술에서부터 20세기 미국의 실험적 미술까지 다루고 있었다. 이처럼 책이 다룬 내용이 방대하기 때문에, 이전부터 관심을 두고 있었던 유럽의 르네상스에 대한 부분을 먼저 읽은 후 나머지 부분을 읽는 방식으로 이 책을 읽어 나갔다. ① 『서양 미술사』는 자료가 풍부하고 해설을 이해하기 어렵지 않아서, 저자가 해설한 내용을 저자의 관점에 따라 받아들이는 것만으로도 충분히 만족스러웠다. 물론 분량이 700여 쪽에 달하는 점은 부담스러웠지만, 하루하루 적당한 분량을 읽도록 계획을 세워서 꾸준히 실천하다 보니 어느새 다 읽었을 만큼 책의 내용은 흥미로웠다.\n</지문>\n<문제>\n윗글을 쓴 학생이 책을 선정할 때 고려한 사항 중, 윗글에서 확인할 수 있는 것은?\n1. 자신의 지식수준에 비추어 적절한 책인가?\n2. 다수의 저자들이 참여하여 집필한 책인가?\n3. 다양한 연령대의 독자에게서 추천받은 책인가?\n4. 이전에 읽은 책과 연관된 내용을 담고 있는 책인가?\n5. 최신의 학술 자료를 활용하여 믿을 만한 내용을 담고 있는 책인가?",
            "answer": "1"
        }
    ]

    # 방법 1: 리스트에서 직접 학습
    train_grpo(
        data_list=example_data,
        model_name="Qwen/Qwen2.5-0.5B-Instruct",  # 테스트용 작은 모델
        output_dir="./grpo_korean_reasoning",
        num_epochs=1,
        batch_size=1,
        num_generations=4,
        max_prompt_length=1024,  # 긴 지문 대응
        max_completion_length=512,  # 추론 과정 포함
        use_lora=True,
    )

    # 방법 2: JSON 파일에서 학습
    # train_grpo(
    #     data_path="./training_data.jsonl",
    #     model_name="Qwen/Qwen2.5-8B-Instruct",
    #     output_dir="./grpo_korean_reasoning",
    #     num_epochs=2,
    #     batch_size=2,
    #     num_generations=8,
    #     use_lora=True,
    # )