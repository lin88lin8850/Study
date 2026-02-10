BK_DIM = 128
MMA_M_DIM = 16
SWIZZLE_BITS_A = 4  # int_log2(BK_dim / 8) with BK_dim=128
SWIZZLE_MASK_A = 0x700 # 0b111 << (SWIZZLE_BITS_A + 4)

def apply_swizzle(lane_id: int) -> int:
    logical_byte_offset = (lane_id % MMA_M_DIM) * BK_DIM * 2
    swizzled_byte_offset = logical_byte_offset ^ ((logical_byte_offset & SWIZZLE_MASK_A) >> SWIZZLE_BITS_A)
    return swizzled_byte_offset

def bank_id(byte_addr: int, word: int = 0) -> int:
    # Shared memory bank width is 4 bytes.
    return ((byte_addr + word * 4) >> 2) & 31

def print_ldmatrix_x2_conflicts():
    print("\n=== ldmatrix.x2 check (active lanes 0..15 => 2 groups x 8 lanes) ===")
    for group in range(2):
        lanes = list(range(group * 8, group * 8 + 8))
        addrs = [apply_swizzle(lane) for lane in lanes]
        print(f"\nGroup {group} lanes: {lanes}")
        for word in range(4):
            banks = [bank_id(addr, word) for addr in addrs]
            unique = len(set(banks))
            conflict = "NO" if unique == 8 else f"YES ({8 - unique} way)"
            print(f"  word{word}: banks={banks}, unique={unique}, conflict={conflict}")


if __name__ == "__main__":
    print_ldmatrix_x2_conflicts()

