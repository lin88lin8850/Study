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

    for i in range(Tr):
        Qi = Q[i * Br : (i + 1) * Br, :]  # [Br, head_dim]
        Oi = O[i * Br : (i + 1) * Br, :]  # [Br, head_dim]
        m_ij_pre = torch.full((Br,1), -torch.inf) # [Br, 1]
        l_ij_pre = torch.zeros((Br, 1)) # [Br, 1]

        for j in range(Tc):
            Kj, Vj = (
                K[j * Bc : (j + 1) * Bc, :],
                V[j * Bc : (j + 1) * Bc, :],
            )  # [Bc, head_dim]

            S_ij = Qi @ Kj.T  # [Br, Bc]
            m_ij = torch.maximum(m_ij_pre, torch.max(S_ij, dim=1).values[:, None])  # [Br, 1]
            P_ij = torch.exp(S_ij - m_ij)  # [Br, Bc]
            l_ij = torch.exp(m_ij_pre-m_ij) * l_ij_pre + torch.sum(P_ij, dim=1)[:, None]  # [Br, 1]
            Oi = torch.exp(m_ij_pre - m_ij) * Oi + P_ij @ Vj

            # update
            m_ij_pre = m_ij
            l_ij_pre = l_ij

        O[i * Br : (i + 1) * Br, :] = Oi / l_ij

    print("flash attention v2 output:\n", O)
    assert(torch.allclose(standard_output, O, rtol=1e-03, atol=1e-05))
