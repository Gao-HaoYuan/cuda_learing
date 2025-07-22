extern "C" __global__
void generate_combinations(int* out) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= 161700) return;

    int remaining = idx;
    int i = 0;

    while (remaining >= ((99 - i) * (98 - i)) / 2) {
        remaining -= ((99 - i) * (98 - i)) / 2;
        ++i;
    }

    int j = i + 1;
    while (remaining >= (99 - j)) {
        remaining -= (99 - j);
        ++j;
    }

    int k = j + 1 + remaining;

    out[3 * idx + 0] = i;
    out[3 * idx + 1] = j;
    out[3 * idx + 2] = k;
}
