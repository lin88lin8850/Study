from sympy import isprime
import numpy as np


max_ngram_size: int = 3
n_head_per_ngram: int = 8
pad_id: int = 2


def find_next_prime(start, seen_primes):
    candidate = start + 1
    while True:
        if isprime(candidate) and candidate not in seen_primes:
            return candidate
        candidate += 1


def calculate_vocab_size_across_layers():
    """
    Calculate vocabulary sizes for n-gram heads across layers
    Returns Example of one layer:
        2-gram vocab_size: [646403, 646411, 646421, 646423, 646433, 646453, 646519, 646523]
        3-gram vocab_size: [646537, 646543, 646549, 646571, 646573, 646577, 646609, 646619]
    """

    layer_id = 1
    vocab_size_per_ngram = [129280 * 5, 129280 * 5]

    seen_primes = set()
    vocab_size_across_layers = {}

    all_ngram_vocab_sizes = []
    for ngram in range(2, max_ngram_size + 1):
        current_ngram_heads_sizes = []

        vocab_size = vocab_size_per_ngram[ngram - 2]
        current_prime_search_start = vocab_size - 1

        for _ in range(n_head_per_ngram):
            found_prime = find_next_prime(current_prime_search_start, seen_primes)
            seen_primes.add(found_prime)
            current_ngram_heads_sizes.append(found_prime)
            current_prime_search_start = found_prime

        all_ngram_vocab_sizes.append(current_ngram_heads_sizes)
    vocab_size_across_layers[layer_id] = all_ngram_vocab_sizes

    return vocab_size_across_layers


def get_layer_multipliers():
    """
    为每一层生成乘数，用于 ngram_hashes 计算，且乘积不会超 int64 上界
    Returns Example:
        {1: array([76993395940407,  4862694818241, 36129212583461]), ...}
    """

    tokenizer_vocab_size = 98627
    PRIME_1 = 10007
    seed = 0
    max_ngram_size = 3
    max_long = np.iinfo(np.int64).max
    M_max = int(max_long // tokenizer_vocab_size)
    half_bound = max(1, M_max // 2)

    layer_multipliers = {}
    layer_ids = [1, 2, 3, 4]

    for layer_id in layer_ids:
        base_seed = int(seed + PRIME_1 * int(layer_id))
        g = np.random.default_rng(base_seed)
        r = g.integers(
            low=0, high=half_bound, size=(max_ngram_size,), dtype=np.int64
        )  # 限制乘积不超 int64 上界
        multipliers = r * 2 + 1  # 保证是奇数, 不丢失低位信息
        layer_multipliers[layer_id] = multipliers

    return layer_multipliers


def get_ngram_hashes(
    input_ids: np.ndarray,
    layer_id: int,
) -> np.ndarray:
    """多头哈希 embedding 架构，用来高效地将变长 n-gram 映射到固定大小的稀疏 embedding 空间"""

    vocab_size_across_layers = calculate_vocab_size_across_layers()
    layer_multipliers = get_layer_multipliers()

    x = np.asarray(input_ids, dtype=np.int64)
    B, T = x.shape

    multipliers = layer_multipliers[layer_id]

    def shift_k(k: int) -> np.ndarray:
        """
        Shift the input array x to the right by k positions, padding with pad_id.
            x: [t0, t1, t2, ..., tT]（原始，当前位置）
            shift_k(1): [pad_id, t0, t1, ..., tT-1] (左移1位, 前1位置)
            shift_k(2): [pad_id, pad_id, t0, t1, ..., tT-2] (左移2位, 前2位置)

        如果取 {x, shift_k(1)}，就可以得到当前位置及其前一个位置的 token 信息，表示 2-gram
        如果取 {x, shift_k(1), shift_k(2)}，就可以得到当前位置及其前两个位置的 token 信息，表示 3-gram
        """
        if k == 0:
            return x
        shifted = np.pad(x, ((0, 0), (k, 0)), mode="constant", constant_values=pad_id)[
            :, :T
        ]
        return shifted

    base_shifts = [shift_k(k) for k in range(max_ngram_size)]

    all_hashes = []

    # 仅计算 2-gram/3-gram hash 值，每个 n-gram 分配 8 个 head
    for n in range(2, max_ngram_size + 1):
        n_gram_index = n - 2
        tokens = base_shifts[:n]  # 取前 n 个 shift 后的 token 矩阵, shape: [n, B, T]

        # 给每个位置赋予位置特定的权重（multipliers[k]），使位置间敏感，即使 token 相同但位置不同，hash 结果也不同
        mix = tokens[0] * multipliers[0]
        for k in range(1, n):
            # xor 操作混合各个 token 信息
            mix = np.bitwise_xor(mix, tokens[k] * multipliers[k])
        num_heads_for_this_ngram = n_head_per_ngram
        head_vocab_sizes = vocab_size_across_layers[layer_id][n_gram_index]

        """
            1. 用质数 head_vocab_size 取模，保证 hash 值均匀分布在 [0, head_vocab_size) 范围内
            2. 每个 head 用不同的质数取模，增加鲁棒性, 防止单个 head 出现碰撞导致无法区分后续的 embedding
        """
        for j in range(num_heads_for_this_ngram):
            mod = int(head_vocab_sizes[j])  # 质数
            head_hash = mix % mod
            all_hashes.append(head_hash.astype(np.int64, copy=False))

    return np.stack(all_hashes, axis=2)


if __name__ == "__main__":

    input_ids = np.array(
        [[0, 1134, 15695, 237, 2049, 1260, 85761, 237, 12071, 36, 9745, 20232, 290, 16]]
    )  # [B, T]

    all_hashes = get_ngram_hashes(
        input_ids, layer_id=1
    )  # [B, T, n_head_per_ngram * (max_ngram_size -1)]=[B, T, 16]
    # print("all_hashes shape:", all_hashes.shape)
    # print("all_hashes:", all_hashes)
