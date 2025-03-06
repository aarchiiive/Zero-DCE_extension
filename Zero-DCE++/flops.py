import torch
from torchprofile import profile_macs  # MACs 계산 라이브러리
import model
from fvcore.nn import FlopCountAnalysis

# 모델 로드
# model = Network()
# model = model.cuda()  # GPU로 이동

scale_factor = 12
DCE_net = model.enhance_net_nopool(scale_factor).cuda()
DCE_net.load_state_dict(torch.load('Exdark/Epoch1.pth'))

# 모델 가중치 로드
# model_path = 'RUAS_Exdark/weights.pt'  # 가중치 경로
# model_dict = torch.load(model_path, map_location='cpu')
# model.load_state_dict(model_dict)

# 입력 크기 정의 (예: 256x256 이미지)
input_size = (720, 1080)  # (높이, 너비)
dummy_input = torch.randn(1, 3, *input_size).cuda()  # 배치 크기 1, 채널 3

# FLOP 계산
flops = FlopCountAnalysis(DCE_net, dummy_input)

# 모델 파라미터 수 계산
param_count = sum(p.numel() for p in DCE_net.parameters())

# GFLOP 계산 (1 GFLOP = 10^9 FLOP)
gflops = flops.total() / 1e9

# 결과 출력
print(f"FLOPs: {flops.total()}")  # 총 FLOP 수 출력
print(f"Parameters: {param_count}")  # 총 파라미터 수 출력
print(f"GFLOPs: {gflops:.6f}")  # GFLOPs 출력
