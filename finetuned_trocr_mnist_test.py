import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torchvision.transforms as transforms
from torchvision import datasets

# 1. 저장된 모델과 프로세서 불러오기
saved_model_path = "./finetuned_trocr_mnist2"
model = VisionEncoderDecoderModel.from_pretrained(saved_model_path)
processor = TrOCRProcessor.from_pretrained(saved_model_path)

# 2. GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 3. MNIST 테스트 데이터셋 준비
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
test_dataset = datasets.MNIST('data', train=False, transform=transform)

# 4. 테스트 함수
def test_model(model, processor, image_tensor, device):
    image = Image.fromarray(image_tensor.squeeze().numpy(), mode='L').convert('RGB')
    image = image.resize((224, 224))
    
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
    
    # 디버깅을 위한 출력 추가
    generated_ids = model.generate(pixel_values)
    print("Generated IDs:", generated_ids)
    
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print("Generated Text:", generated_text)
    
    return generated_text

# 5. 테스트 실행
model.eval()
with torch.no_grad():
    for i in range(10):
        image, true_label = test_dataset[i]
        predicted_text = test_model(model, processor, image, device)
        print(f"실제 숫자: {true_label}, 예측된 숫자: {predicted_text}")
