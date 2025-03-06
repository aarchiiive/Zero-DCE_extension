import torch
import model

# 모델 로드 및 weights 적용
scale_factor = 12
DCE_net = model.enhance_net_nopool(scale_factor).cuda()
DCE_net.load_state_dict(torch.load('Exdark/Epoch1.pth'))

# 입력 크기 정의 (예: 720x1080 이미지, 배치 크기 1, 채널 3)
input_size = (720, 1080)
dummy_input = torch.randn(1, 3, *input_size).cuda()

# GPU warm-up: 초기 오버헤드를 줄이기 위해 몇 번 실행
for _ in range(10):
    _ = DCE_net(dummy_input)

# 100번의 추론 실행에 대해 시간 측정
times = []
for _ in range(100):
    torch.cuda.synchronize()  # 이전 작업들이 모두 끝났는지 동기화
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()     # 시작 시간 기록
    _ = DCE_net(dummy_input) # 추론 실행
    end_event.record()       # 종료 시간 기록

    torch.cuda.synchronize() # 모든 GPU 연산이 완료될 때까지 대기
    elapsed_time_ms = start_event.elapsed_time(end_event)  # 경과 시간 (밀리초 단위)
    times.append(elapsed_time_ms)

# 100번 실행에 대한 평균 추론 시간 계산
avg_time = sum(times) / len(times)
print("Average inference time over 100 runs: {:.3f} ms".format(avg_time))
