#이 파일은 해설지를 첨부하면 해설지에 있는 각 문항 별 답안을 json파일로 추출하는 코드입니다.


import anthropic
import fitz # PyMuPDF
import json


def extract_text_from_pdf(pdf_path):
  """PDF 파일에서 텍스트 추출"""
  text = ""
  with fitz.open(pdf_path) as pdf:
      for page in pdf:
          text += page.get_text()
  return text


def split_text_into_chunks(text, chunk_size=3000):
  """텍스트를 더 작은 청크로 나누기"""
  return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]


def get_answers_from_gpt(pdf_text, api_key):
  """Claude를 사용하여 PDF 텍스트에서 답안 추출"""
  client = anthropic.Anthropic(api_key=api_key)
  chunks = split_text_into_chunks(pdf_text)
  all_answers = {}
 
  for i, chunk in enumerate(chunks, 1):
      print(f"청크 {i}/{len(chunks)} 처리 중...")
     
      try:
          response = client.messages.create(
              model="claude-3-opus-20240229",
              messages=[{
                  "role": "user",
                  "content": f"""다음은 해설지의 텍스트입니다. 각 문항별 정답을 JSON 형식으로 추출하세요.
                  형식은 {{"1번": 정답, "2번": 정답, ...}}입니다.
                  정답만 추출하고 다른 설명은 제외하세요.
                  만약 답안이 ④와 같은 형태라면 4라고 써주세요.
                  모든 경우가 그렇다고 할 수는 없지만, 보통 정답은 '정답'이라고 써져있는 텍스트 바로 뒤에 나오는 숫자가 답인 경우가 많으니 그걸 참고해서 정확도를 신경써주세요.
                  시간은 더 걸려도 좋으니 정확도에 무조건 더 신경써서 정확도 높은 결과를 주세요. 지금 정확도가 낮으니 정확도 신경써주세요. 만약, 답안을 따로 앞에 두는 경우도 있으니 그게 있다면 그걸 참고해서 하는게 좋을 것 같아요.
                 
                  해설지 텍스트:
                  {chunk}"""
              }],
              max_tokens=1000
          )
         
          # TextBlock 객체에서 텍스트 추출
          if hasattr(response.content[0], 'text'):
              content = response.content[0].text
          else:
              content = str(response.content[0])
         
          # JSON 문자열에서 작은따옴표를 큰따옴표로 변경
          content = content.replace("'", '"')
         
          # JSON 파싱
          chunk_answers = json.loads(content)
         
          # 번호 형식 통일 (키에 "번" 추가)
          formatted_answers = {}
          for key, value in chunk_answers.items():
              if "번" not in str(key):
                  formatted_key = f"{key}번"
              else:
                  formatted_key = str(key)
              formatted_answers[formatted_key] = value
             
          all_answers.update(formatted_answers)
         
      except json.JSONDecodeError as e:
          print(f"청크 {i}에서 JSON 파싱 오류 발생: {e}")
          print("응답:", content)
          continue
      except Exception as e:
          print(f"청크 {i}에서 오류 발생: {str(e)}")
          continue
 
  return all_answers


def save_answers_to_json(answers, output_path):
  """추출된 답안을 JSON 파일로 저장"""
  with open(output_path, "w", encoding="utf-8") as file:
      json.dump(answers, file, ensure_ascii=False, indent=4)


def main():
  # 경로와 API 키 설정
  pdf_path = "./physics.pdf"
  output_path = "output_answers.json"
  api_key = "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"
 
  try:
      # PDF에서 텍스트 추출
      print("PDF에서 텍스트 추출 중...")
      pdf_text = extract_text_from_pdf(pdf_path)
     
      # Claude를 사용하여 답안 추출
      print("Claude를 사용하여 답안 추출 중...")
      answers = get_answers_from_gpt(pdf_text, api_key)
     
      # 결과를 JSON 파일로 저장
      print("결과를 JSON 파일로 저장 중...")
      save_answers_to_json(answers, output_path)
     
      print(f"답안이 추출되어 {output_path}에 저장되었습니다.")
     
  except Exception as e:
      print(f"오류 발생: {str(e)}")


if __name__ == "__main__":
  main()
