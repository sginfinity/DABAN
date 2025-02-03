import openai
import base64
import json
import re  # 정규식 사용을 위한 import

# OpenAI API 키 설정
client = openai.OpenAI(api_key="GPT_API_KEY")

# 이미지 파일 읽기 및 Base64 인코딩
image_path = "student_answer05.jpg"
with open(image_path, "rb") as image_file:
    image_data = base64.b64encode(image_file.read()).decode('utf-8')

# API 호출
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "Extract numbers from the provided image and also extract the printed code located at the bottom-right corner. Return them as structured JSON."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Please identify the numbers from this image and the printed code in the bottom-right corner. Return them in JSON format with question numbers as keys and the code as '코드'."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
            ]
        }
    ],
    temperature=0
)

# 결과 저장
output_filename = "output_result.json"

try:
    # 1️⃣ 응답에서 content 추출
    content = response.choices[0].message.content

    # 2️⃣ 마크다운 코드 블록 제거
    clean_content = re.sub(r"```json|```", "", content).strip()

    # 3️⃣ JSON 객체 부분만 추출
    json_match = re.search(r"{.*}", clean_content, re.DOTALL)  # 중괄호로 시작해서 끝나는 부분 추출
    if json_match:
        json_data = json_match.group()

        # 4️⃣ JSON 파싱
        extracted_data = json.loads(json_data)

        # 5️⃣ JSON을 보기 좋은 형식으로 변환 후 출력 (파일 저장과 동일한 방식)
        formatted_json = json.dumps(extracted_data, ensure_ascii=False, indent=4)
        print(formatted_json)  # 🔹 터미널에 파일과 동일한 JSON 출력

        # 6️⃣ JSON 파일로 저장
        with open(output_filename, "w", encoding="utf-8") as json_file:
            json_file.write(formatted_json)

        print(f"결과가 '{output_filename}' 파일로 저장되었습니다.")

    else:
        raise ValueError("JSON 데이터가 발견되지 않았습니다.")

except (json.JSONDecodeError, ValueError) as e:
    # JSON 파싱 실패 시 원본 응답 저장
    with open(output_filename, "w", encoding="utf-8") as json_file:
        json.dump(response.to_dict(), json_file, ensure_ascii=False, indent=4)

    print(f"JSON 파싱에 실패했습니다. 원본 응답이 '{output_filename}' 파일로 저장되었습니다.")
    print("에러 메시지:", e)
