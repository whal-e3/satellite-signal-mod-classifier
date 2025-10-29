#!/usr/bin/env python3
import json
import os
from openai import OpenAI
from tqdm import tqdm

JSONL_PATH = 'amc_finetune_dataset0722.jsonl' 
OPENAI_ORG_ID = None

try:
    # 조직 ID가 지정된 경우, 클라이언트 초기화 시 함께 전달합니다.
    client = OpenAI(organization=OPENAI_ORG_ID)
    print(f"✅ OpenAI 클라이언트 초기화 완료.")
    if OPENAI_ORG_ID:
        print(f"   - 조직 ID: {OPENAI_ORG_ID}")
    else:
        print(f"   - 조직 ID: 기본값 사용")

except Exception as e:
    print(f"OpenAI 클라이언트 초기화 오류: {e}")
    print("OPENAI_API_KEY 환경 변수가 올바르게 설정되었는지 확인해주세요.")
    exit()

def validate_fine_tuning_data(jsonl_path):
    """
    JSONL 파일을 분석하여 참조된 이미지 파일들이 현재 OpenAI 계정 및 조직에
    올바르게 존재하는지, purpose가 'vision'인지 검증합니다.
    """
    print("="*60)
    print("OpenAI 파인튜닝 데이터 검증을 시작합니다.")
    print(f"대상 파일: {jsonl_path}")
    print("="*60)

    # 1. JSONL 파일 존재 여부 확인
    if not os.path.exists(jsonl_path):
        print(f"[오류] 파일을 찾을 수 없습니다: {jsonl_path}")
        return

    # 2. JSONL 파일에서 필요한 파일 ID 목록 추출
    required_file_ids = set()
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                for msg in data.get("messages", []):
                    if isinstance(msg.get("content"), list):
                        for part in msg["content"]:
                            if part.get("type") == "image_url":
                                url = part.get("image_url", {}).get("url", "")
                                if url.startswith("openai://"):
                                    file_id = url.replace("openai://", "")
                                    required_file_ids.add(file_id)
    except Exception as e:
        print(f"[오류] JSONL 파일을 읽는 중 문제가 발생했습니다: {e}")
        return

    if not required_file_ids:
        print("[정보] JSONL 파일에서 'openai://' 형식의 이미지 참조를 찾지 못했습니다.")
        return

    print(f"\n📄 JSONL 파일에서 총 {len(required_file_ids)}개의 고유한 파일 ID를 발견했습니다.")

    # 3. 현재 API 키 및 조직 ID로 접근 가능한 모든 파일 목록 가져오기
    print("\n🌐 현재 API 키 및 조직 ID로 접근 가능한 모든 파일 목록을 OpenAI 서버에서 가져오는 중...")
    try:
        all_files_list = list(client.files.list())
        accessible_files = {file.id: file for file in all_files_list}
        print(f"✅ 총 {len(accessible_files)}개의 파일을 찾았습니다.")
    except Exception as e:
        print(f"[치명적 오류] OpenAI에서 파일 목록을 가져오는 데 실패했습니다: {e}")
        return

    # 4. 파일 ID 대조 및 검증
    print("\n🔬 파일 ID를 하나씩 대조하여 검증합니다...")
    
    found_files = set()
    missing_files = set()
    wrong_purpose_files = {}

    for file_id in tqdm(required_file_ids, desc="파일 검증 중"):
        if file_id in accessible_files:
            found_files.add(file_id)
            file_obj = accessible_files[file_id]
            if file_obj.purpose != 'vision':
                wrong_purpose_files[file_id] = file_obj.purpose
        else:
            missing_files.add(file_id)

    # 5. 최종 결과 보고
    print("\n--- 최종 검증 결과 ---")

    if not missing_files and not wrong_purpose_files:
        print("\n[성공] 🎉 모든 것이 완벽합니다!")
        print(f"JSONL 파일에 참조된 {len(found_files)}개의 파일 모두 현재 계정/조직에 존재하며, 'purpose'가 'vision'으로 올바르게 설정되어 있습니다.")
        print("\n[다음 단계] 파인튜닝 작업을 생성할 때도 반드시 동일한 조직 ID를 지정해주세요!")
    else:
        if found_files:
            print(f"\n[확인] ✅ {len(found_files)}개의 파일은 정상적으로 확인되었습니다.")

        if missing_files:
            print(f"\n[오류] 🚨 {len(missing_files)}개의 파일을 현재 API 키/조직 ID로 찾을 수 없습니다!")
            print("이 파일들은 다른 조직 ID로 업로드되었거나, 삭제되었거나, 파일 ID에 오타가 있을 수 있습니다.")
            print("아래 목록을 확인해주세요:")
            for i, file_id in enumerate(missing_files):
                print(f"  - {file_id}")

        if wrong_purpose_files:
            print(f"\n[오류] 🚨 {len(wrong_purpose_files)}개의 파일의 'purpose' 설정이 잘못되었습니다!")
            print("파인튜닝에 사용될 이미지는 반드시 'purpose'가 'vision'이어야 합니다.")
            print("아래 목록을 확인해주세요:")
            for file_id, purpose in wrong_purpose_files.items():
                print(f"  - 파일 ID: {file_id}, 현재 purpose: '{purpose}' (올바른 값: 'vision')")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    if JSONL_PATH == 'final_finetune_data.jsonl':
         print("="*50)
         print("‼️  스크립트를 실행하기 전에 상단의 'JSONL_PATH'와 'OPENAI_ORG_ID' 변수를")
         print("   실제 사용하실 값으로 수정해주세요.")
         print("="*50)
    else:
        validate_fine_tuning_data(JSONL_PATH)