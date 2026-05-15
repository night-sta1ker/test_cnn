def transform(A, W=4):
    m = len(A)
    n = len(A[0])

    used = {}

    # 关键：按原矩阵行优先扫描
    for i in range(m):
        for j in range(n):
            r = i + j
            c = j % W

            while (r, c) in used:
                r += 1
                c = (c + 1) % W

            used[(r, c)] = (A[i][j], i, j)

    max_r = max(r for r, c in used)
    B = [[0 for _ in range(W)] for _ in range(max_r + 1)]
    SRC = [[None for _ in range(W)] for _ in range(max_r + 1)]

    for (r, c), (val, i, j) in used.items():
        B[r][c] = val
        SRC[r][c] = (i, j)

    return B, SRC


A = [
    [1,  2,  3,  4,  5],
    [6,  7,  8,  9,  10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25],
]

B, SRC = transform(A)

print("B:")
for row in B:
    print(row)

print("\nSRC:")
for r, row in enumerate(SRC):
    print(r, row)