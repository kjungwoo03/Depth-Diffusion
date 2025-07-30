import os
import sys
from PIL import Image
import matplotlib.pyplot as plt
import depth_pro


def depth_pro_inference(image_path, model, transform):
    image, _, f_px = depth_pro.load_rgb(image_path)
    image = transform(image)
    prediction = model.infer(image, f_px=f_px)
    depth = prediction["depth"]  # Depth in [m].
    focallength_px = prediction["focallength_px"]  # Focal length in pixels.

    return depth, focallength_px


def main():
    image_path = "results/generated_42.png"
    model, transform = depth_pro.create_model_and_transforms()
    model.eval()
    depth, focallength_px = depth_pro_inference(image_path, model, transform)
    
    viz=True
    if viz:
        # 원본 이미지 로드
        original_image = Image.open(image_path)
        
        # depth map을 시각화를 위해 numpy 배열로 변환
        depth_np = depth.squeeze().cpu().numpy()
        

        plt.figure(figsize=(12, 6))
        
        # 원본 이미지 크기 조정
        original_image = original_image.resize((depth_np.shape[1], depth_np.shape[0]))
        
        # 원본 이미지 표시
        plt.subplot(1, 2, 1)
        plt.imshow(original_image)
        plt.title('Original Image')
        plt.axis('off')
        
        # depth map 표시
        plt.subplot(1, 2, 2)
        plt.imshow(depth_np, cmap='plasma')
        plt.colorbar(label='Depth (meter)', shrink=0.4)
        plt.title(f'Depth Map (Focal length: {focallength_px:.2f}px)')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('results/depth_comparison.png')
        print("Depth map saved to results/depth_comparison.png")
        plt.close()


if __name__ == "__main__":
    main()