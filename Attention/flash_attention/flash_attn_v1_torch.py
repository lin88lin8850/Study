import torch
import torch.nn.functional as F

torch.manual_seed(0)


def standard_attention(Q, K, V):
    P = Q @ K.T
    S = F.softmax(P, dim=1)
    out = S @ V

    return out


if __name__ == "__main__":
    N, head_dim = 1024, 64

    Q = torch.randn(N, head_dim)
    K = torch.randn(N, head_dim)
    V = torch.randn(N, head_dim)

    standard_output = standard_attention(Q, K, V)
    print("standard attention output:\n", standard_output)

    # flash attention v1 torch implementation
    Bc, Br = 16, 16
    Tc, Tr = N // Bc, N // Br

    # O, m, l initialization
    O = torch.zeros(N, head_dim)
    m = torch.full((N,1), -torch.inf)
    l = torch.zeros((N, 1))

    for j in range(Tc):
        Kj, Vj = (
            K[j * Bc : (j + 1) * Bc, :],
            V[j * Bc : (j + 1) * Bc, :],
        )  # [Bc, head_dim]

        for i in range(Tr):
            mi = m[i * Br : (i + 1) * Br, :] #[Br, 1]
            li = l[i * Br : (i + 1) * Br, :] #[ Br, 1]
            Qi = Q[i * Br : (i + 1) * Br, :]  # [Br, head_dim]
            Oi = O[i * Br : (i + 1) * Br, :]  # [Br, head_dim]

            S_ij = Qi @ Kj.T  # [Br, Bc]
            m_ij = torch.max(S_ij, dim=1).values[:, None] # [Br, 1]            
            P_ij = torch.exp(S_ij - m_ij)  # [Br, Bc]
            l_ij = torch.sum(P_ij, dim=1)[:, None]  # [Br, 1]

            # update
            mi_new = torch.maximum(mi, m_ij)  # [Br, 1]
            li_new = torch.exp(mi - mi_new) * li + torch.exp(m_ij - mi_new) * l_ij # [Br, 1]
            Oi = li * torch.exp(mi - mi_new) * Oi / li_new + (torch.exp(m_ij - mi_new) * P_ij / li_new) @ Vj

            m[i * Br : (i + 1) * Br, :] = mi_new
            l[i * Br : (i + 1) * Br, :] = li_new
            O[i * Br : (i + 1) * Br, :] = Oi
    
    print("flash attention output:\n", O)
    assert(torch.allclose(standard_output, O, rtol=1e-03, atol=1e-05))
