# Depth-Diffusion

Stable Diffusion을 활용한 이미지 생성 및 어텐션 맵 시각화 프로젝트입니다.

## 기능

- Stable Diffusion을 사용한 텍스트-이미지 생성
- Cross-Attention 맵 시각화
- 토큰별 어텐션 진화 과정 분석

## 설치 방법

필요한 패키지 설치:

```bash
pip install torch diffusers transformers pillow matplotlib tqdm
```

## 사용 방법

기본적인 이미지 생성 및 어텐션 맵 시각화:

```bash
python p2p-inference.py --prompt "A photo of a cat wearing a space suit" --num_inference_steps 50 --guidance_scale 7.5
```

### 매개변수

- `--prompt`: 생성할 이미지 설명 (문자열)
- `--num_inference_steps`: 생성 단계 수 (기본값: 50)
- `--guidance_scale`: 분류기 가이던스 스케일 (기본값: 7.5)
- `--seed`: 랜덤 시드 (기본값: 42)

## 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요. 