import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, VisionEncoderDecoderConfig
from PIL import Image
import numpy as np
from tqdm import tqdm
import os

# 메모리 관리 설정
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# 1. MNIST 데이터 준비
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

mnist_train = datasets.MNIST('data', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST('data', train=False, transform=transform)

# 2. 데이터 증강 및 TrOCR 형식으로 변환
class AugmentedMNISTDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def augment_image(self, image):
        # 데이터 증강 로직
        image = transforms.RandomRotation(10)(image)
        image = transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))(image)
        return image

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = self.augment_image(image)
        
        # 이미지 처리
        image = Image.fromarray(image.squeeze().numpy(), mode='L').convert('RGB')
        image = image.resize((224, 224))
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()
        
        # 레이블 처리 수정
        label_str = str(label)
        encoded = self.processor.tokenizer(
            label_str,
            padding="max_length",
            max_length=8,
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze()
        
        return {"pixel_values": pixel_values, "labels": encoded}
    
# 3. TrOCR 모델 및 프로세서 로드
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# 모델 설정 업데이트
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size
model.config.eos_token_id = processor.tokenizer.sep_token_id  # 추가
model.config.max_length = 8  # 추가

print(f"decoder_start_token_id: {model.config.decoder_start_token_id}")
print(f"pad_token_id: {model.config.pad_token_id}")
print(f"vocab_size: {model.config.vocab_size}")

# 디코더 설정 적용
model.decoder.config.decoder_start_token_id = model.config.decoder_start_token_id
model.decoder.config.pad_token_id = model.config.pad_token_id

# 4. 데이터셋 및 데이터 로더 생성
train_dataset = AugmentedMNISTDataset(mnist_train, processor)
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# validation 데이터셋과 데이터로더 추가
val_dataset = AugmentedMNISTDataset(mnist_test, processor)
val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False)

# 5. 학습 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)  # 학습률 수정

# validation 함수
def validate(model, val_loader, device):
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            
            if pixel_values.dim() == 3:
                pixel_values = pixel_values.unsqueeze(0)
            if labels.dim() == 1:
                labels = labels.unsqueeze(0)
                
            outputs = model(pixel_values=pixel_values, labels=labels)
            total_val_loss += outputs.loss.item()
    return total_val_loss / len(val_loader)

# 6. 학습 루프
num_epochs = 5  # 에폭 수 증가
accumulation_steps = 4

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch+1}/{num_epochs}")
    
    for batch_idx, batch in progress_bar:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        if pixel_values.dim() == 3:
            pixel_values = pixel_values.unsqueeze(0)
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)

        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss / accumulation_steps
        total_loss += loss.item() * accumulation_steps

        loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        progress_bar.set_postfix({'loss': f"{loss.item() * accumulation_steps:.4f}"})

        if batch_idx % 100 == 0:
            torch.cuda.empty_cache()

    if len(train_dataloader) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    avg_loss = total_loss / len(train_dataloader)
    val_loss = validate(model, val_dataloader, device)  # validation 추가
    
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Average Training Loss: {avg_loss:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Learning Rate: {optimizer.param_groups[0]['lr']}")

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': avg_loss,
        'val_loss': val_loss,  # validation loss 저장 추가
    }, f'checkpoint_epoch_{epoch}.pth')

model.save_pretrained("./finetuned_trocr_mnist2")
processor.save_pretrained("./finetuned_trocr_mnist2")

print("Fine-tuning complete. Model saved.")
