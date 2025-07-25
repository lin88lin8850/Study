import torch

s1 = torch.cuda.Stream()
s2 = torch.cuda.Stream()


A = torch.randn(2048, 2048, device='cuda')
B = torch.randn(2048, 2048, device='cuda')

torch.cuda.synchronize()

with torch.cuda.stream(s1):
    C = torch.mm(A, A)
with torch.cuda.stream(s2):
    D = torch.mm(B, B)

torch.cuda.synchronize()