import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
from tqdm import tqdm
import random
import os

# CUDA 메모리 관리 설정
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.cuda.empty_cache()

# 기본 설정
random.seed(42)
torch.manual_seed(42)

# 이미지 크기 설정
IMAGE_SIZE = (128, 128)

# 데이터 준비
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

mnist_train = datasets.MNIST('data', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST('data', train=False, transform=transform)

# 작은 데이터셋 생성
train_indices = random.sample(range(len(mnist_train)), 100)
test_indices = random.sample(range(len(mnist_test)), 20)

train_subset = Subset(mnist_train, train_indices)
test_subset = Subset(mnist_test, test_indices)

class QuickMNISTDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        
        # 이미지 처리
        image = Image.fromarray(image.squeeze().numpy(), mode='L').convert('RGB')
        image = image.resize(IMAGE_SIZE)
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()
        
        # 레이블 처리
        label_str = str(label)
        # 시작 토큰과 끝 토큰을 추가
        label_str = f"{self.processor.tokenizer.bos_token}{label_str}{self.processor.tokenizer.eos_token}"
        
        encoded = self.processor.tokenizer(
            label_str,
            padding='max_length',
            max_length=8,
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze()
        
        return {"pixel_values": pixel_values, "labels": encoded}

# 모델과 프로세서 로드
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# 모델 설정 업데이트
model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.eos_token_id = processor.tokenizer.eos_token_id
model.config.max_length = 8

print(f"Tokenizer special tokens: {processor.tokenizer.special_tokens_map}")
print(f"Vocab size: {len(processor.tokenizer)}")
print(f"BOS token id: {processor.tokenizer.bos_token_id}")
print(f"EOS token id: {processor.tokenizer.eos_token_id}")
print(f"PAD token id: {processor.tokenizer.pad_token_id}")

# 데이터 로더 설정
train_dataset = QuickMNISTDataset(train_subset, processor)
test_dataset = QuickMNISTDataset(test_subset, processor)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

# 학습 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

def quick_test(model, processor, test_loader, device, num_samples=5):
    model.eval()
    torch.cuda.empty_cache()
    
    with torch.no_grad():
        for batch in test_loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"]
            
            outputs = model.generate(
                pixel_values,
                max_length=8,
                num_beams=2,
                early_stopping=True,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                bos_token_id=processor.tokenizer.bos_token_id,
                use_cache=True
            )
            
            print("Generated token IDs shape:", outputs.shape)
            print("Generated token IDs:", outputs)
            
            # 원본 토큰 ID 출력
            print("\nLabel token IDs:", labels)
            
            # 디코딩된 결과
            predicted_texts = processor.batch_decode(outputs, skip_special_tokens=True)
            true_texts = processor.batch_decode(labels, skip_special_tokens=True)
            
            for pred, true in zip(predicted_texts, true_texts):
                print(f"예측: {pred}, 실제: {true}")
            
            num_samples -= len(predicted_texts)
            if num_samples <= 0:
                break
            
            torch.cuda.empty_cache()

# 학습 루프
num_epochs = 1

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    
    for batch_idx, batch in enumerate(progress_bar):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        if (batch_idx + 1) % 20 == 0:
            print(f"\nBatch {batch_idx+1}")
            print(f"Average Loss: {total_loss / (batch_idx + 1):.4f}")
            print("\n현재 학습 상황 테스트:")
            quick_test(model, processor, test_loader, device)
            model.train()
        
        if batch_idx % 5 == 0:
            torch.cuda.empty_cache()

print("\n최종 테스트:")
quick_test(model, processor, test_loader, device, num_samples=10)
