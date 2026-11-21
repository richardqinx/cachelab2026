/*
请注意，你的代码不能出现任何 int/short/char/float/double/auto 等局部变量/函数传参，我们仅允许使用 reg 定义的寄存器变量。
其中 reg 等价于一个 int。

你不能自己申请额外的内存，即不能使用 new/malloc，作为补偿我们传入了一段 buffer，大小为 BUFFER_SIZE = 64，你可以视情况使用。

我们的数组按照 A, B, C, buffer 的顺序在内存上连续紧密排列，且 &A = 0x30000000（这是模拟的设定，不是 A 的真实地址）

如果你需要以更自由的方式访问内存，你可以以相对 A 的方式访问，比如 A[100]，用 *(0x30000000) 是无法访问到的。

如果你有定义常量的需求（更严谨的说法是，你想定义的是汇编层面的立即数，不应该占用寄存器），请参考下面这种方式使用宏定义来完成。
*/

#include "cachelab.h"

#define m case0_m
#define n case0_n
#define p case0_p

// 我们用这个 2*2*2 的矩阵乘法来演示寄存器是怎么被分配的
void gemm_case0(ptr_reg A, ptr_reg B, ptr_reg C, ptr_reg buffer)
{ // allocate 0 1 2 3
  for (reg i = 0; i < m; ++i)
  { // allocate 4
    for (reg j = 0; j < p; ++j)
    {               // allocate 5
      reg tmpc = 0; // allocate 6
      for (reg k = 0; k < n; ++k)
      {                          // allocate 7
        reg tmpa = A[i * n + k]; // allocate 8
        reg tmpb = B[k * p + j]; // allocate 9
        tmpc += tmpa * tmpb;
      } // free 9 8
      // free 7
      C[i * p + j] = tmpc;
    } // free 6
    // free 5
  }
  // free 4
} // free 3 2 1 0

#undef m
#undef n
#undef p

// Case 1 (16x16x16): 6x4 / 4x4 寄存器分块 + 任意 K 起点循环 + 同行 A 预读
#define CASE1_K(START, STEP, T) (((START) + (STEP) * (T)) % 16)

#define CASE1_PAIR(ROWS, SAVE_ROW, ROW, COL, START, STEP)                      \
  do {                                                                         \
    reg acc[(ROWS) * 4];                                                       \
    for (reg t = 0; t < 16; t += 2) {                                          \
      reg k = CASE1_K(START, STEP, t);                                         \
      reg vals[4];                                                             \
      vals[0] = B[k * 16 + (COL) + 0];                                         \
      vals[1] = B[k * 16 + (COL) + 1];                                         \
      vals[2] = B[k * 16 + (COL) + 2];                                         \
      vals[3] = B[k * 16 + (COL) + 3];                                         \
      reg ahead;                                                               \
      reg temp;                                                                \
      for (reg r = 0; r < (ROWS); ++r) {                                       \
        temp = A[((ROW) + r) * 16 + k];                                        \
        acc[r * 4 + 0] += temp * vals[0];                                      \
        acc[r * 4 + 1] += temp * vals[1];                                      \
        acc[r * 4 + 2] += temp * vals[2];                                      \
        acc[r * 4 + 3] += temp * vals[3];                                      \
        if (r == (SAVE_ROW) &&                                                 \
            k / 4 == CASE1_K(START, STEP, t + 1) / 4)                          \
          ahead = A[((ROW) + r) * 16 + CASE1_K(START, STEP, t + 1)];           \
      }                                                                        \
      vals[0] = B[CASE1_K(START, STEP, t + 1) * 16 + (COL) + 0];               \
      vals[1] = B[CASE1_K(START, STEP, t + 1) * 16 + (COL) + 1];               \
      vals[2] = B[CASE1_K(START, STEP, t + 1) * 16 + (COL) + 2];               \
      vals[3] = B[CASE1_K(START, STEP, t + 1) * 16 + (COL) + 3];               \
      for (reg r = 0; r < (ROWS); ++r) {                                       \
        if (r == (SAVE_ROW) &&                                                 \
            k / 4 == CASE1_K(START, STEP, t + 1) / 4)                          \
          temp = ahead;                                                        \
        else                                                                   \
          temp = A[((ROW) + r) * 16 + CASE1_K(START, STEP, t + 1)];            \
        acc[r * 4 + 0] += temp * vals[0];                                      \
        acc[r * 4 + 1] += temp * vals[1];                                      \
        acc[r * 4 + 2] += temp * vals[2];                                      \
        acc[r * 4 + 3] += temp * vals[3];                                      \
      }                                                                        \
    }                                                                          \
    for (reg r = 0; r < (ROWS); ++r) {                                         \
      for (reg c = 0; c < 4; ++c) {                                            \
        C[((ROW) + r) * 16 + (COL) + c] = acc[r * 4 + c];                      \
      }                                                                        \
    }                                                                          \
  } while (false)

void gemm_case1(ptr_reg A, ptr_reg B, ptr_reg C, ptr_reg buffer) {
  { ptr_reg unused = std::move(buffer); } // 释放 1 个指针寄存器，给行循环 r
  // (ROWS, SAVE_ROW, ROW, COL, START, STEP)
  CASE1_PAIR(6,  0,  0,  0,  3, 15);
  CASE1_PAIR(6,  3,  0,  4,  7, 15);
  CASE1_PAIR(6,  3,  0,  8,  6,  1);
  CASE1_PAIR(6,  3,  0, 12, 15, 15);
  CASE1_PAIR(6,  1,  6,  0,  4,  1);
  CASE1_PAIR(6,  1,  6,  4,  4,  1);
  CASE1_PAIR(6,  2,  6,  8, 11, 15);
  CASE1_PAIR(6,  1,  6, 12, 12,  1);
  CASE1_PAIR(4, -1, 12,  0,  9,  1);
  CASE1_PAIR(4,  3, 12,  4,  0,  1);
  CASE1_PAIR(4, -1, 12,  8,  8,  1);
  CASE1_PAIR(4,  0, 12, 12,  1, 15);
}

#undef CASE1_PAIR
#undef CASE1_K

// Case 2 (32x32x32): 4x4 寄存器分块 + K 从 col 起循环 + 第 2 行 A 预读
void gemm_case2(ptr_reg A, ptr_reg B, ptr_reg C, ptr_reg buffer) {
  for (reg row = 0; row < 32; row += 4) {
    for (reg col = 0; col < 32; col += 4) {
      reg acc[16];
      for (reg t = 0; t < 32; t += 2) {
        reg k = (col + t) % 32;
        reg vals[4];
        vals[0] = A[(row + 0) * 32 + k];
        vals[1] = A[(row + 1) * 32 + k];
        vals[2] = A[(row + 2) * 32 + k];
        vals[3] = A[(row + 3) * 32 + k];
        reg ahead;
        if (k / 4 == ((col + t + 1) % 32) / 4)
          ahead = A[(row + 2) * 32 + ((col + t + 1) % 32)];
        for (reg c = 0; c < 4; ++c) {
          reg temp = B[k * 32 + col + c];
          acc[0 * 4 + c] += vals[0] * temp;
          acc[1 * 4 + c] += vals[1] * temp;
          acc[2 * 4 + c] += vals[2] * temp;
          acc[3 * 4 + c] += vals[3] * temp;
        }
        vals[0] = A[(row + 0) * 32 + ((col + t + 1) % 32)];
        vals[1] = A[(row + 1) * 32 + ((col + t + 1) % 32)];
        if (k / 4 == ((col + t + 1) % 32) / 4)
          vals[2] = ahead;
        else
          vals[2] = A[(row + 2) * 32 + ((col + t + 1) % 32)];
        vals[3] = A[(row + 3) * 32 + ((col + t + 1) % 32)];
        for (reg c = 0; c < 4; ++c) {
          reg temp = B[((col + t + 1) % 32) * 32 + col + c];
          acc[0 * 4 + c] += vals[0] * temp;
          acc[1 * 4 + c] += vals[1] * temp;
          acc[2 * 4 + c] += vals[2] * temp;
          acc[3 * 4 + c] += vals[3] * temp;
        }
      }
      for (reg r = 0; r < 4; ++r) {
        for (reg c = 0; c < 4; ++c) {
          C[(row + r) * 32 + col + c] = acc[r * 4 + c];
        }
      }
    }
  }
}

// Case 3 (27x41x27): 4x7/3x7/4x6/3x6 蛇形分块 + 任意 K 起点 + 同行 A 预读
#define CASE3_B_OFF (case3_m * case3_n)
#define CASE3_C_OFF (CASE3_B_OFF + case3_n * case3_p)
#define CASE3_K(START, STEP, T) (((START) + case3_n + (STEP) * (T)) % case3_n)
#define CASE3_A_IDX(ROW, U, K) (((ROW) + (U)) * case3_n + (K))
#define CASE3_SAME_LINE(ROW, U, K, NEXT)                                       \
  (CASE3_A_IDX(ROW, U, K) / 4 == CASE3_A_IDX(ROW, U, NEXT) / 4)

#define CASE3_MUL_COL(ROWS, COLS, COL, K, C_IDX)                               \
  do {                                                                         \
    b = A[CASE3_B_OFF + (K) * case3_p + (COL) + (C_IDX)];                      \
    sum[0 * (COLS) + (C_IDX)] += a[0] * b;                                     \
    sum[1 * (COLS) + (C_IDX)] += a[1] * b;                                     \
    sum[2 * (COLS) + (C_IDX)] += a[2] * b;                                     \
    if ((ROWS) == 4)                                                           \
      sum[((ROWS) == 4 ? 3 : 0) * (COLS) + (C_IDX)] +=                         \
          a[((ROWS) == 4 ? 3 : 0)] * b;                                        \
  } while (false)

#define CASE3_PRODUCTS(ROWS, COLS, COL, K)                                     \
  do {                                                                         \
    reg b;                                                                     \
    CASE3_MUL_COL(ROWS, COLS, COL, K, 0);                                      \
    CASE3_MUL_COL(ROWS, COLS, COL, K, 1);                                      \
    CASE3_MUL_COL(ROWS, COLS, COL, K, 2);                                      \
    CASE3_MUL_COL(ROWS, COLS, COL, K, 3);                                      \
    CASE3_MUL_COL(ROWS, COLS, COL, K, 4);                                      \
    CASE3_MUL_COL(ROWS, COLS, COL, K, 5);                                      \
    if ((COLS) == 7)                                                           \
      CASE3_MUL_COL(ROWS, COLS, COL, K, ((COLS) == 7 ? 6 : 0));                \
  } while (false)

#define CASE3_LOAD_A(ROW, U, K, K_NEXT, SAVE, SLOT)                            \
  do {                                                                         \
    a[U] = A[CASE3_A_IDX(ROW, U, K)];                                          \
    if (t + 1 < case3_n && (SAVE) && CASE3_SAME_LINE(ROW, U, K, K_NEXT))       \
      ahead[SLOT] = A[CASE3_A_IDX(ROW, U, K_NEXT)];                            \
  } while (false)

#define CASE3_NEXT_A(ROW, U, K, K_NEXT, SAVE, SLOT)                            \
  do {                                                                         \
    if ((SAVE) && CASE3_SAME_LINE(ROW, U, K, K_NEXT))                          \
      a[U] = ahead[SLOT];                                                      \
    else                                                                       \
      a[U] = A[CASE3_A_IDX(ROW, U, K_NEXT)];                                   \
  } while (false)

#define CASE3_PAIR(ROWS, COLS, ROW, COL, START, STEP, COUNT, S0, S1, S2, S3)   \
  do {                                                                         \
    reg sum[(ROWS) * (COLS)];                                                  \
    for (reg t = 0; t < case3_n; t += 2) {                                     \
      reg a[4];                                                                \
      reg ahead[COUNT];                                                        \
      CASE3_LOAD_A(ROW, 0, CASE3_K(START, STEP, t),                            \
                   CASE3_K(START, STEP, t + 1), S0, 0);                        \
      CASE3_LOAD_A(ROW, 1, CASE3_K(START, STEP, t),                            \
                   CASE3_K(START, STEP, t + 1), S1, (S0));                     \
      CASE3_LOAD_A(ROW, 2, CASE3_K(START, STEP, t),                            \
                   CASE3_K(START, STEP, t + 1), S2, (S0) + (S1));              \
      if ((ROWS) == 4)                                                         \
        CASE3_LOAD_A(ROW, 3, CASE3_K(START, STEP, t),                          \
                     CASE3_K(START, STEP, t + 1), S3, (S0) + (S1) + (S2));     \
      CASE3_PRODUCTS(ROWS, COLS, COL, CASE3_K(START, STEP, t));                \
      if (t + 1 < case3_n) {                                                   \
        CASE3_NEXT_A(ROW, 0, CASE3_K(START, STEP, t),                          \
                     CASE3_K(START, STEP, t + 1), S0, 0);                      \
        CASE3_NEXT_A(ROW, 1, CASE3_K(START, STEP, t),                          \
                     CASE3_K(START, STEP, t + 1), S1, (S0));                   \
        CASE3_NEXT_A(ROW, 2, CASE3_K(START, STEP, t),                          \
                     CASE3_K(START, STEP, t + 1), S2, (S0) + (S1));            \
        if ((ROWS) == 4)                                                       \
          CASE3_NEXT_A(ROW, 3, CASE3_K(START, STEP, t),                        \
                       CASE3_K(START, STEP, t + 1), S3, (S0) + (S1) + (S2));   \
        CASE3_PRODUCTS(ROWS, COLS, COL, CASE3_K(START, STEP, t + 1));          \
      }                                                                        \
    }                                                                          \
    for (reg r = 0; r < (ROWS); ++r) {                                         \
      for (reg c = 0; c < (COLS); ++c) {                                       \
        A[CASE3_C_OFF + ((ROW) + r) * case3_p + (COL) + c] =                   \
            sum[r * (COLS) + c];                                               \
      }                                                                        \
    }                                                                          \
  } while (false)

void gemm_case3(ptr_reg A, ptr_reg B, ptr_reg C, ptr_reg buffer) {
  {
    ptr_reg unused_buffer = std::move(buffer);
    ptr_reg unused_B = std::move(B);
    ptr_reg unused_C = std::move(C);
  }
  // (ROWS, COLS, ROW, COL, START, STEP, COUNT, S0, S1, S2, S3)
  CASE3_PAIR(4, 7,  0,  0, 38, -1, 1, 0, 1, 0, 0);
  CASE3_PAIR(4, 7,  4,  0, 40,  1, 1, 0, 1, 0, 0);
  CASE3_PAIR(4, 7,  8,  0, 30,  1, 1, 1, 0, 0, 0);
  CASE3_PAIR(4, 7, 12,  0, 33, -1, 1, 1, 0, 0, 0);
  CASE3_PAIR(4, 7, 16,  0, 36,  1, 1, 0, 0, 0, 1);
  CASE3_PAIR(4, 7, 20,  0, 22,  1, 1, 1, 0, 0, 0);
  CASE3_PAIR(3, 7, 24,  0, 21, -1, 2, 1, 1, 0, 0);
  CASE3_PAIR(3, 7, 24,  7, 22,  1, 1, 0, 0, 0, 0);
  CASE3_PAIR(4, 7, 20,  7, 22, -1, 1, 1, 0, 0, 0);
  CASE3_PAIR(4, 7, 16,  7, 23,  1, 1, 0, 0, 1, 0);
  CASE3_PAIR(4, 7, 12,  7, 19, -1, 1, 0, 0, 0, 1);
  CASE3_PAIR(4, 7,  8,  7, 18,  1, 1, 0, 0, 1, 0);
  CASE3_PAIR(4, 7,  4,  7, 17, -1, 1, 0, 1, 0, 0);
  CASE3_PAIR(4, 7,  0,  7, 18,  1, 1, 1, 0, 0, 0);
  CASE3_PAIR(4, 7,  0, 14, 17, -1, 1, 1, 0, 0, 0);
  CASE3_PAIR(4, 7,  4, 14, 18,  1, 1, 1, 0, 0, 0);
  CASE3_PAIR(4, 7,  8, 14, 17, -1, 1, 0, 1, 0, 0);
  CASE3_PAIR(4, 7, 12, 14, 18,  1, 1, 0, 0, 0, 1);
  CASE3_PAIR(4, 7, 16, 14, 17, -1, 1, 0, 1, 0, 0);
  CASE3_PAIR(4, 7, 20, 14, 19,  1, 1, 0, 0, 1, 0);
  CASE3_PAIR(3, 7, 24, 14, 18, -1, 2, 1, 1, 0, 0);
  CASE3_PAIR(3, 6, 24, 21, 22,  1, 3, 1, 1, 1, 0);
  CASE3_PAIR(4, 6, 20, 21, 21, -1, 3, 1, 1, 1, 0);
  CASE3_PAIR(4, 6, 16, 21, 22,  1, 3, 1, 1, 0, 1);
  CASE3_PAIR(4, 6, 12, 21, 17, -1, 2, 1, 1, 0, 0);
  CASE3_PAIR(4, 6,  8, 21, 14,  1, 2, 1, 1, 0, 0);
  CASE3_PAIR(4, 6,  4, 21, 10, -1, 3, 1, 0, 1, 1);
  CASE3_PAIR(4, 6,  0, 21, 11,  1, 3, 0, 1, 1, 1);
}

#undef CASE3_PAIR
#undef CASE3_NEXT_A
#undef CASE3_LOAD_A
#undef CASE3_PRODUCTS
#undef CASE3_MUL_COL
#undef CASE3_SAME_LINE
#undef CASE3_A_IDX
#undef CASE3_K
#undef CASE3_C_OFF
#undef CASE3_B_OFF
