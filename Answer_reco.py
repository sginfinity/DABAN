#이 파일은 학생이 작성한 답안지를 인식해 각 문항답 학생이 작성한 답안을 json 파일 형태로 저장하는 코드입니다.


import fitz # PyMuPDF
from PIL import Image, ImageDraw
import base64
from io import BytesIO
import json
from anthropic import Anthropic


def encode_image_to_base64(image_path):
  with Image.open(image_path) as image:
      if image.mode == 'RGBA':
          image = image.convert('RGB')
      buffered = BytesIO()
      image.save(buffered, format="JPEG")
      img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
      return img_str


def analyze_handwritten_numbers(base64_image, api_key):
  client = Anthropic(api_key=api_key)
 
  response = client.messages.create(
      model="claude-3-opus-20240229",
      messages=[{
          "role": "user",
          "content": [
              {
                  "type": "image",
                  "source": {
                      "type": "base64",
                      "media_type": "image/jpeg",
                      "data": base64_image
                  }
              },
              {
                  "type": "text",
                  "text": "이 이미지에서 빨간 박스 안에 쓰여진 필기체 숫자만 알려주세요. 다른 설명 없이 숫자만 출력해주세요. 시간이 더 걸려도 되니까 정확한 인식 부탁합니다.그리고 숫자 앞뒤 '따옴표'는 결과값에 출력하지 말아주세요"
                 
              }
          ]
      }],
      max_tokens=300
  )
 
  if hasattr(response, 'content') and isinstance(response.content, str):
      return response.content
  elif hasattr(response, 'content'):
      return str(response.content[0]) if response.content else "No content"
  else:
      return str(response)


def process_pdf_and_analyze(pdf_path, output_image_path, api_key):
  doc = fitz.open(pdf_path)
  page = doc[0] #이거는 수정해야함. 지금은 파일이 길어서 첫번째 페이지만 인식하도록 함.
  empty_boxes = []
  pix = page.get_pixmap()
  img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
  draw = ImageDraw.Draw(img)
 
  tables = page.find_tables()
  for table in tables:
      for cell in table.cells:
          rect = fitz.Rect(cell)
          text = page.get_text(clip=rect, sort=True).strip()
          if not text:
              empty_boxes.append({
                  'x0': rect.x0,
                  'y0': rect.y0,
                  'x1': rect.x1,
                  'y1': rect.y1,
                  'width': rect.width,
                  'height': rect.height
              })
              draw.rectangle(
                  [(rect.x0, rect.y0), (rect.x1, rect.y1)],
                  outline="red",
                  width=2
              )
 
  img.save(output_image_path)
  doc.close()
  print(f"이미지가 {output_image_path}에 저장되었습니다.")
 
  # 분석 결과 저장
  results = {}
  for i, box in enumerate(empty_boxes, 1):
      # 박스 이미지를 잘라내기
      cropped_image = img.crop((box['x0'], box['y0'], box['x1'], box['y1']))
      # 잘라낸 이미지를 base64로 인코딩
      buffered = BytesIO()
      cropped_image.save(buffered, format="JPEG")
      base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
     
      try:
          result = analyze_handwritten_numbers(base64_image, api_key)
          if isinstance(result, str):
              # 숫자만 추출하여 저장
              number_only = ''.join(filter(str.isdigit, result.strip()))
              results[f"{i}번"] = number_only if number_only else result.strip()
          else:
              results[f"{i}번"] = str(result)
      except Exception as e:
          print(f"박스 {i} 분석 중 오류 발생: {e}")
          results[f"{i}번"] = f"Error: {str(e)}"
 
  # JSON 파일로 저장
  json_output_path = "answers.json"  # 파일명을 명확하게 지정
  with open(json_output_path, 'w', encoding='utf-8') as f:
      json.dump(results, f, ensure_ascii=False, indent=4)
 
  print(f"답안이 JSON 파일로 저장되었습니다: {json_output_path}")
  return results


# 사용 예
if __name__ == "__main__":
  pdf_path = "./test1.pdf"
  output_image_path = "./output_image.png"
  api_key = "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"
 
  results = process_pdf_and_analyze(pdf_path, output_image_path, api_key)
  # JSON 결과 출력
  print(json.dumps(results, indent=4, ensure_ascii=False))
