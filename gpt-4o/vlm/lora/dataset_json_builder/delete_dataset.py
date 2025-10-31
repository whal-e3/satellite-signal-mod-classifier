#!/usr/bin/env python3
import os
import openai

# === CONFIG ===
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    print("오류: OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
    exit()

openai.api_key = API_KEY

def delete_png_files_only():
    """
    OpenAI 계정의 모든 파일 중 .png 파일만 확인 절차 없이 즉시 삭제합니다.
    """
    try:
        print("OpenAI 계정의 모든 파일을 조회합니다...")
        all_files = openai.files.list()
        
        if not all_files.data:
            print("삭제할 파일이 없습니다.")
            return

        print(f"총 {len(all_files.data)}개의 파일을 찾았습니다. 이 중 .png 파일만 삭제합니다.")
        print("❗ 경고: 확인 절차 없이 즉시 .png 파일 삭제를 시작합니다.")
        
        deleted_count = 0
        skipped_count = 0

        for file_obj in all_files.data:
            # CHANGED: 파일명이 .png로 끝나는지 확인 (대소문자 무시)
            if file_obj.filename.lower().endswith('.png'):
                try:
                    print(f"  - PNG 파일 삭제 중: {file_obj.filename} (ID: {file_obj.id})")
                    status = openai.files.delete(file_obj.id)
                    if status.deleted:
                        print(f"    ✓ 성공적으로 삭제되었습니다.")
                        deleted_count += 1
                    else:
                        print(f"    ✗ 삭제에 실패했습니다. 상태: {status}")
                except Exception as e:
                    print(f"    ✗ 파일 {file_obj.id} 삭제 중 오류 발생: {e}")
            else:
                # .png 파일이 아니면 건너뜀
                print(f"  - 건너뜀 (PNG 아님): {file_obj.filename}")
                skipped_count += 1
        
        print("\n✅ 모든 파일 확인 작업이 완료되었습니다.")
        print(f"총 {deleted_count}개의 PNG 파일을 삭제했고, {skipped_count}개의 파일을 건너뛰었습니다.")

    except Exception as e:
        print(f"API 호출 중 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    delete_png_files_only()