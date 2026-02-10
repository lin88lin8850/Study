def apply_swizzle(lane_id: int, smem_stride: int = 32) -> int:
    logical_offset = (lane_id % 32) * smem_stride  # in half elements
    swizzled_offset = logical_offset ^ ((logical_offset & 0b10000000) >> 4)
    swizzled_offset = swizzled_offset ^ ((swizzled_offset & 0b1100000) >> 2)
    return swizzled_offset  # still in half elements


def bank_id_from_half_index(half_index: int, word: int = 0) -> int:
    # shared memory bank is 4B wide. half index -> byte address = half_index * 2.
    byte_addr = half_index * 2 + word * 4
    return (byte_addr >> 2) & 31


print("=== Per-lane mapping (correct unit conversion) ===")
for lane_id in range(32):
    new_offset = apply_swizzle(lane_id)
    bank0 = bank_id_from_half_index(new_offset, 0)
    print(
        f"lane={lane_id:2d}, half_idx={new_offset:3d}, "
        f"bank(word0)={bank0:2d}"
    )

print("\n=== ldmatrix-style check: 4 groups x 8 lanes, 4 words/row ===")
for group in range(4):
    lanes = list(range(group * 8, group * 8 + 8))
    offsets = [apply_swizzle(lane) for lane in lanes]
    print(f"\nGroup {group} lanes: {lanes}")
    for word in range(4):
        banks = [bank_id_from_half_index(off, word) for off in offsets]
        unique = len(set(banks))
        conflict = "NO" if unique == 8 else f"YES ({8 - unique} way)"
        print(f"  word{word}: banks={banks}, unique={unique}, conflict={conflict}")
