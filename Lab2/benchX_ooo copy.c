#define N 40

int main()
{
    int A[N][N];
    int B[N][N];
    int C[N][N] = {};

    register int i, j, k, *aa, *bb, *aa1, *bb1, *aa2, *bb2, *aa3, *bb3, *aa4, *bb4, *aa5, *bb5, *aa6, *bb6, *aa7, *bb7, *cc, temp, s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, i;
    for (i = 0; i < N; i+=8)
    {
        aa = A[i];
        bb = B[i];
        aa1 = A[i+1];
        bb1 = B[i+1];
        aa2 = A[i+2];
        bb2 = B[i+2];
        aa3 = A[i+3];
        bb3 = B[i+3];
        aa4 = A[i+4];
        bb4 = B[i+4];
        aa5 = A[i+5];
        bb5 = B[i+5];
        aa6 = A[i+6];
        bb6 = B[i+6];
        aa7 = A[i+7];
        bb7 = B[i+7];
        aa[0] = i;
        aa[1] = i + 1;
        aa[2] = i + 2;
        aa[3] = i + 3;
        aa[4] = i + 4;
        aa[5] = i + 5;
        aa[6] = i + 6;
        aa[7] = i + 7;
        aa[8] = i + 8;
        aa[9] = i + 9;
        aa[10] =   i + 10;
        aa[11] =   i + 11;
        aa[12] =   i + 12;
        aa[13] =   i + 13;
        aa[14] =   i + 14;
        aa[15] =   i + 15;
        aa[16] =   i + 16;
        aa[17] =   i + 17;
        aa[18] =   i + 18;
        aa[19] =   i + 19;
        aa[20] =   i + 20;
        aa[21] =   i + 21;
        aa[22] =   i + 22;
        aa[23] =   i + 23;
        aa[24] =   i + 24;
        aa[25] =   i + 25;
        aa[26] =   i + 26;
        aa[27] =   i + 27;
        aa[28] =   i + 28;
        aa[29] =   i + 29;
        aa[30] =   i + 30;
        aa[31] =   i + 31;
        aa[32] =   i + 32;
        aa[33] =   i + 33;
        aa[34] =   i + 34;
        aa[35] =   i + 35;
        aa[36] =   i + 36;
        aa[37] =   i + 37;
        aa[38] =   i + 38;
        aa[39] =   i + 39;
        bb[0] = i;
        bb[1] = i - 1;
        bb[2] = i - 2;
        bb[3] = i - 3;
        bb[4] = i - 4;
        bb[5] = i - 5;
        bb[6] = i - 6;
        bb[7] = i - 7;
        bb[8] = i - 8;
        bb[9] = i - 9;
        bb[10] =    i - 10;
        bb[11] =    i - 11;
        bb[12] =    i - 12;
        bb[13] =    i - 13;
        bb[14] =    i - 14;
        bb[15] =    i - 15;
        bb[16] =    i - 16;
        bb[17] =    i - 17;
        bb[18] =    i - 18;
        bb[19] =    i - 19;
        bb[20] =    i - 20;
        bb[21] =    i - 21;
        bb[22] =    i - 22;
        bb[23] =    i - 23;
        bb[24] =    i - 24;
        bb[25] =    i - 25;
        bb[26] =    i - 26;
        bb[27] =    i - 27;
        bb[28] =    i - 28;
        bb[29] =    i - 29;
        bb[30] =    i - 30;
        bb[31] =    i - 31;
        bb[32] =    i - 32;
        bb[33] =    i - 33;
        bb[34] =    i - 34;
        bb[35] =    i - 35;
        bb[36] =    i - 36;
        bb[37] =    i - 37;
        bb[38] =    i - 38;
        bb[39] =    i - 39;
        i++;
        aa1[0] = i;
        aa1[1] = i + 1;
        aa1[2] = i + 2;
        aa1[3] = i + 3;
        aa1[4] = i + 4;
        aa1[5] = i + 5;
        aa1[6] = i + 6;
        aa1[7] = i + 7;
        aa1[8] = i + 8;
        aa1[9] = i + 9;
        aa1[10] =   i + 10;
        aa1[11] =   i + 11;
        aa1[12] =   i + 12;
        aa1[13] =   i + 13;
        aa1[14] =   i + 14;
        aa1[15] =   i + 15;
        aa1[16] =   i + 16;
        aa1[17] =   i + 17;
        aa1[18] =   i + 18;
        aa1[19] =   i + 19;
        aa1[20] =   i + 20;
        aa1[21] =   i + 21;
        aa1[22] =   i + 22;
        aa1[23] =   i + 23;
        aa1[24] =   i + 24;
        aa1[25] =   i + 25;
        aa1[26] =   i + 26;
        aa1[27] =   i + 27;
        aa1[28] =   i + 28;
        aa1[29] =   i + 29;
        aa1[30] =   i + 30;
        aa1[31] =   i + 31;
        aa1[32] =   i + 32;
        aa1[33] =   i + 33;
        aa1[34] =   i + 34;
        aa1[35] =   i + 35;
        aa1[36] =   i + 36;
        aa1[37] =   i + 37;
        aa1[38] =   i + 38;
        aa1[39] =   i + 39;
        bb1[0] = i;
        bb1[1] = i - 1;
        bb1[2] = i - 2;
        bb1[3] = i - 3;
        bb1[4] = i - 4;
        bb1[5] = i - 5;
        bb1[6] = i - 6;
        bb1[7] = i - 7;
        bb1[8] = i - 8;
        bb1[9] = i - 9;
        bb1[10] =    i - 10;
        bb1[11] =    i - 11;
        bb1[12] =    i - 12;
        bb1[13] =    i - 13;
        bb1[14] =    i - 14;
        bb1[15] =    i - 15;
        bb1[16] =    i - 16;
        bb1[17] =    i - 17;
        bb1[18] =    i - 18;
        bb1[19] =    i - 19;
        bb1[20] =    i - 20;
        bb1[21] =    i - 21;
        bb1[22] =    i - 22;
        bb1[23] =    i - 23;
        bb1[24] =    i - 24;
        bb1[25] =    i - 25;
        bb1[26] =    i - 26;
        bb1[27] =    i - 27;
        bb1[28] =    i - 28;
        bb1[29] =    i - 29;
        bb1[30] =    i - 30;
        bb1[31] =    i - 31;
        bb1[32] =    i - 32;
        bb1[33] =    i - 33;
        bb1[34] =    i - 34;
        bb1[35] =    i - 35;
        bb1[36] =    i - 36;
        bb1[37] =    i - 37;
        bb1[38] =    i - 38;
        bb1[39] =    i - 39;
        i++;
        aa2[0] = i;
        aa2[1] = i + 1;
        aa2[2] = i + 2;
        aa2[3] = i + 3;
        aa2[4] = i + 4;
        aa2[5] = i + 5;
        aa2[6] = i + 6;
        aa2[7] = i + 7;
        aa2[8] = i + 8;
        aa2[9] = i + 9;
        aa2[10] =   i + 10;
        aa2[11] =   i + 11;
        aa2[12] =   i + 12;
        aa2[13] =   i + 13;
        aa2[14] =   i + 14;
        aa2[15] =   i + 15;
        aa2[16] =   i + 16;
        aa2[17] =   i + 17;
        aa2[18] =   i + 18;
        aa2[19] =   i + 19;
        aa2[20] =   i + 20;
        aa2[21] =   i + 21;
        aa2[22] =   i + 22;
        aa2[23] =   i + 23;
        aa2[24] =   i + 24;
        aa2[25] =   i + 25;
        aa2[26] =   i + 26;
        aa2[27] =   i + 27;
        aa2[28] =   i + 28;
        aa2[29] =   i + 29;
        aa2[30] =   i + 30;
        aa2[31] =   i + 31;
        aa2[32] =   i + 32;
        aa2[33] =   i + 33;
        aa2[34] =   i + 34;
        aa2[35] =   i + 35;
        aa2[36] =   i + 36;
        aa2[37] =   i + 37;
        aa2[38] =   i + 38;
        aa2[39] =   i + 39;
        bb2[0] = i;
        bb2[1] = i - 1;
        bb2[2] = i - 2;
        bb2[3] = i - 3;
        bb2[4] = i - 4;
        bb2[5] = i - 5;
        bb2[6] = i - 6;
        bb2[7] = i - 7;
        bb2[8] = i - 8;
        bb2[9] = i - 9;
        bb2[10] =    i - 10;
        bb2[11] =    i - 11;
        bb2[12] =    i - 12;
        bb2[13] =    i - 13;
        bb2[14] =    i - 14;
        bb2[15] =    i - 15;
        bb2[16] =    i - 16;
        bb2[17] =    i - 17;
        bb2[18] =    i - 18;
        bb2[19] =    i - 19;
        bb2[20] =    i - 20;
        bb2[21] =    i - 21;
        bb2[22] =    i - 22;
        bb2[23] =    i - 23;
        bb2[24] =    i - 24;
        bb2[25] =    i - 25;
        bb2[26] =    i - 26;
        bb2[27] =    i - 27;
        bb2[28] =    i - 28;
        bb2[29] =    i - 29;
        bb2[30] =    i - 30;
        bb2[31] =    i - 31;
        bb2[32] =    i - 32;
        bb2[33] =    i - 33;
        bb2[34] =    i - 34;
        bb2[35] =    i - 35;
        bb2[36] =    i - 36;
        bb2[37] =    i - 37;
        bb2[38] =    i - 38;
        bb2[39] =    i - 39;
        i++;
        aa3[0] = i;
        aa3[1] = i + 1;
        aa3[2] = i + 2;
        aa3[3] = i + 3;
        aa3[4] = i + 4;
        aa3[5] = i + 5;
        aa3[6] = i + 6;
        aa3[7] = i + 7;
        aa3[8] = i + 8;
        aa3[9] = i + 9;
        aa3[10] =   i + 10;
        aa3[11] =   i + 11;
        aa3[12] =   i + 12;
        aa3[13] =   i + 13;
        aa3[14] =   i + 14;
        aa3[15] =   i + 15;
        aa3[16] =   i + 16;
        aa3[17] =   i + 17;
        aa3[18] =   i + 18;
        aa3[19] =   i + 19;
        aa3[20] =   i + 20;
        aa3[21] =   i + 21;
        aa3[22] =   i + 22;
        aa3[23] =   i + 23;
        aa3[24] =   i + 24;
        aa3[25] =   i + 25;
        aa3[26] =   i + 26;
        aa3[27] =   i + 27;
        aa3[28] =   i + 28;
        aa3[29] =   i + 29;
        aa3[30] =   i + 30;
        aa3[31] =   i + 31;
        aa3[32] =   i + 32;
        aa3[33] =   i + 33;
        aa3[34] =   i + 34;
        aa3[35] =   i + 35;
        aa3[36] =   i + 36;
        aa3[37] =   i + 37;
        aa3[38] =   i + 38;
        aa3[39] =   i + 39;
        bb3[0] = i;
        bb3[1] = i - 1;
        bb3[2] = i - 2;
        bb3[3] = i - 3;
        bb3[4] = i - 4;
        bb3[5] = i - 5;
        bb3[6] = i - 6;
        bb3[7] = i - 7;
        bb3[8] = i - 8;
        bb3[9] = i - 9;
        bb3[10] =    i - 10;
        bb3[11] =    i - 11;
        bb3[12] =    i - 12;
        bb3[13] =    i - 13;
        bb3[14] =    i - 14;
        bb3[15] =    i - 15;
        bb3[16] =    i - 16;
        bb3[17] =    i - 17;
        bb3[18] =    i - 18;
        bb3[19] =    i - 19;
        bb3[20] =    i - 20;
        bb3[21] =    i - 21;
        bb3[22] =    i - 22;
        bb3[23] =    i - 23;
        bb3[24] =    i - 24;
        bb3[25] =    i - 25;
        bb3[26] =    i - 26;
        bb3[27] =    i - 27;
        bb3[28] =    i - 28;
        bb3[29] =    i - 29;
        bb3[30] =    i - 30;
        bb3[31] =    i - 31;
        bb3[32] =    i - 32;
        bb3[33] =    i - 33;
        bb3[34] =    i - 34;
        bb3[35] =    i - 35;
        bb3[36] =    i - 36;
        bb3[37] =    i - 37;
        bb3[38] =    i - 38;
        bb3[39] =    i - 39;
        i++;
        aa4[0] = i;
        aa4[1] = i + 1;
        aa4[2] = i + 2;
        aa4[3] = i + 3;
        aa4[4] = i + 4;
        aa4[5] = i + 5;
        aa4[6] = i + 6;
        aa4[7] = i + 7;
        aa4[8] = i + 8;
        aa4[9] = i + 9;
        aa4[10] =   i + 10;
        aa4[11] =   i + 11;
        aa4[12] =   i + 12;
        aa4[13] =   i + 13;
        aa4[14] =   i + 14;
        aa4[15] =   i + 15;
        aa4[16] =   i + 16;
        aa4[17] =   i + 17;
        aa4[18] =   i + 18;
        aa4[19] =   i + 19;
        aa4[20] =   i + 20;
        aa4[21] =   i + 21;
        aa4[22] =   i + 22;
        aa4[23] =   i + 23;
        aa4[24] =   i + 24;
        aa4[25] =   i + 25;
        aa4[26] =   i + 26;
        aa4[27] =   i + 27;
        aa4[28] =   i + 28;
        aa4[29] =   i + 29;
        aa4[30] =   i + 30;
        aa4[31] =   i + 31;
        aa4[32] =   i + 32;
        aa4[33] =   i + 33;
        aa4[34] =   i + 34;
        aa4[35] =   i + 35;
        aa4[36] =   i + 36;
        aa4[37] =   i + 37;
        aa4[38] =   i + 38;
        aa4[39] =   i + 39;
        bb4[0] = i;
        bb4[1] = i - 1;
        bb4[2] = i - 2;
        bb4[3] = i - 3;
        bb4[4] = i - 4;
        bb4[5] = i - 5;
        bb4[6] = i - 6;
        bb4[7] = i - 7;
        bb4[8] = i - 8;
        bb4[9] = i - 9;
        bb4[10] =    i - 10;
        bb4[11] =    i - 11;
        bb4[12] =    i - 12;
        bb4[13] =    i - 13;
        bb4[14] =    i - 14;
        bb4[15] =    i - 15;
        bb4[16] =    i - 16;
        bb4[17] =    i - 17;
        bb4[18] =    i - 18;
        bb4[19] =    i - 19;
        bb4[20] =    i - 20;
        bb4[21] =    i - 21;
        bb4[22] =    i - 22;
        bb4[23] =    i - 23;
        bb4[24] =    i - 24;
        bb4[25] =    i - 25;
        bb4[26] =    i - 26;
        bb4[27] =    i - 27;
        bb4[28] =    i - 28;
        bb4[29] =    i - 29;
        bb4[30] =    i - 30;
        bb4[31] =    i - 31;
        bb4[32] =    i - 32;
        bb4[33] =    i - 33;
        bb4[34] =    i - 34;
        bb4[35] =    i - 35;
        bb4[36] =    i - 36;
        bb4[37] =    i - 37;
        bb4[38] =    i - 38;
        bb4[39] =    i - 39;
        i++;
        aa5[0] = i;
        aa5[1] = i + 1;
        aa5[2] = i + 2;
        aa5[3] = i + 3;
        aa5[4] = i + 4;
        aa5[5] = i + 5;
        aa5[6] = i + 6;
        aa5[7] = i + 7;
        aa5[8] = i + 8;
        aa5[9] = i + 9;
        aa5[10] =   i + 10;
        aa5[11] =   i + 11;
        aa5[12] =   i + 12;
        aa5[13] =   i + 13;
        aa5[14] =   i + 14;
        aa5[15] =   i + 15;
        aa5[16] =   i + 16;
        aa5[17] =   i + 17;
        aa5[18] =   i + 18;
        aa5[19] =   i + 19;
        aa5[20] =   i + 20;
        aa5[21] =   i + 21;
        aa5[22] =   i + 22;
        aa5[23] =   i + 23;
        aa5[24] =   i + 24;
        aa5[25] =   i + 25;
        aa5[26] =   i + 26;
        aa5[27] =   i + 27;
        aa5[28] =   i + 28;
        aa5[29] =   i + 29;
        aa5[30] =   i + 30;
        aa5[31] =   i + 31;
        aa5[32] =   i + 32;
        aa5[33] =   i + 33;
        aa5[34] =   i + 34;
        aa5[35] =   i + 35;
        aa5[36] =   i + 36;
        aa5[37] =   i + 37;
        aa5[38] =   i + 38;
        aa5[39] =   i + 39;
        bb5[0] = i;
        bb5[1] = i - 1;
        bb5[2] = i - 2;
        bb5[3] = i - 3;
        bb5[4] = i - 4;
        bb5[5] = i - 5;
        bb5[6] = i - 6;
        bb5[7] = i - 7;
        bb5[8] = i - 8;
        bb5[9] = i - 9;
        bb5[10] =    i - 10;
        bb5[11] =    i - 11;
        bb5[12] =    i - 12;
        bb5[13] =    i - 13;
        bb5[14] =    i - 14;
        bb5[15] =    i - 15;
        bb5[16] =    i - 16;
        bb5[17] =    i - 17;
        bb5[18] =    i - 18;
        bb5[19] =    i - 19;
        bb5[20] =    i - 20;
        bb5[21] =    i - 21;
        bb5[22] =    i - 22;
        bb5[23] =    i - 23;
        bb5[24] =    i - 24;
        bb5[25] =    i - 25;
        bb5[26] =    i - 26;
        bb5[27] =    i - 27;
        bb5[28] =    i - 28;
        bb5[29] =    i - 29;
        bb5[30] =    i - 30;
        bb5[31] =    i - 31;
        bb5[32] =    i - 32;
        bb5[33] =    i - 33;
        bb5[34] =    i - 34;
        bb5[35] =    i - 35;
        bb5[36] =    i - 36;
        bb5[37] =    i - 37;
        bb5[38] =    i - 38;
        bb5[39] =    i - 39;
        i++;
        aa6[0] = i;
        aa6[1] = i + 1;
        aa6[2] = i + 2;
        aa6[3] = i + 3;
        aa6[4] = i + 4;
        aa6[5] = i + 5;
        aa6[6] = i + 6;
        aa6[7] = i + 7;
        aa6[8] = i + 8;
        aa6[9] = i + 9;
        aa6[10] =   i + 10;
        aa6[11] =   i + 11;
        aa6[12] =   i + 12;
        aa6[13] =   i + 13;
        aa6[14] =   i + 14;
        aa6[15] =   i + 15;
        aa6[16] =   i + 16;
        aa6[17] =   i + 17;
        aa6[18] =   i + 18;
        aa6[19] =   i + 19;
        aa6[20] =   i + 20;
        aa6[21] =   i + 21;
        aa6[22] =   i + 22;
        aa6[23] =   i + 23;
        aa6[24] =   i + 24;
        aa6[25] =   i + 25;
        aa6[26] =   i + 26;
        aa6[27] =   i + 27;
        aa6[28] =   i + 28;
        aa6[29] =   i + 29;
        aa6[30] =   i + 30;
        aa6[31] =   i + 31;
        aa6[32] =   i + 32;
        aa6[33] =   i + 33;
        aa6[34] =   i + 34;
        aa6[35] =   i + 35;
        aa6[36] =   i + 36;
        aa6[37] =   i + 37;
        aa6[38] =   i + 38;
        aa6[39] =   i + 39;
        bb6[0] = i;
        bb6[1] = i - 1;
        bb6[2] = i - 2;
        bb6[3] = i - 3;
        bb6[4] = i - 4;
        bb6[5] = i - 5;
        bb6[6] = i - 6;
        bb6[7] = i - 7;
        bb6[8] = i - 8;
        bb6[9] = i - 9;
        bb6[10] =    i - 10;
        bb6[11] =    i - 11;
        bb6[12] =    i - 12;
        bb6[13] =    i - 13;
        bb6[14] =    i - 14;
        bb6[15] =    i - 15;
        bb6[16] =    i - 16;
        bb6[17] =    i - 17;
        bb6[18] =    i - 18;
        bb6[19] =    i - 19;
        bb6[20] =    i - 20;
        bb6[21] =    i - 21;
        bb6[22] =    i - 22;
        bb6[23] =    i - 23;
        bb6[24] =    i - 24;
        bb6[25] =    i - 25;
        bb6[26] =    i - 26;
        bb6[27] =    i - 27;
        bb6[28] =    i - 28;
        bb6[29] =    i - 29;
        bb6[30] =    i - 30;
        bb6[31] =    i - 31;
        bb6[32] =    i - 32;
        bb6[33] =    i - 33;
        bb6[34] =    i - 34;
        bb6[35] =    i - 35;
        bb6[36] =    i - 36;
        bb6[37] =    i - 37;
        bb6[38] =    i - 38;
        bb6[39] =    i - 39;
        i++;
        aa7[0] = i;
        aa7[1] = i + 1;
        aa7[2] = i + 2;
        aa7[3] = i + 3;
        aa7[4] = i + 4;
        aa7[5] = i + 5;
        aa7[6] = i + 6;
        aa7[7] = i + 7;
        aa7[8] = i + 8;
        aa7[9] = i + 9;
        aa7[10] =   i + 10;
        aa7[11] =   i + 11;
        aa7[12] =   i + 12;
        aa7[13] =   i + 13;
        aa7[14] =   i + 14;
        aa7[15] =   i + 15;
        aa7[16] =   i + 16;
        aa7[17] =   i + 17;
        aa7[18] =   i + 18;
        aa7[19] =   i + 19;
        aa7[20] =   i + 20;
        aa7[21] =   i + 21;
        aa7[22] =   i + 22;
        aa7[23] =   i + 23;
        aa7[24] =   i + 24;
        aa7[25] =   i + 25;
        aa7[26] =   i + 26;
        aa7[27] =   i + 27;
        aa7[28] =   i + 28;
        aa7[29] =   i + 29;
        aa7[30] =   i + 30;
        aa7[31] =   i + 31;
        aa7[32] =   i + 32;
        aa7[33] =   i + 33;
        aa7[34] =   i + 34;
        aa7[35] =   i + 35;
        aa7[36] =   i + 36;
        aa7[37] =   i + 37;
        aa7[38] =   i + 38;
        aa7[39] =   i + 39;
        bb7[0] = i;
        bb7[1] = i - 1;
        bb7[2] = i - 2;
        bb7[3] = i - 3;
        bb7[4] = i - 4;
        bb7[5] = i - 5;
        bb7[6] = i - 6;
        bb7[7] = i - 7;
        bb7[8] = i - 8;
        bb7[9] = i - 9;
        bb7[10] =    i - 10;
        bb7[11] =    i - 11;
        bb7[12] =    i - 12;
        bb7[13] =    i - 13;
        bb7[14] =    i - 14;
        bb7[15] =    i - 15;
        bb7[16] =    i - 16;
        bb7[17] =    i - 17;
        bb7[18] =    i - 18;
        bb7[19] =    i - 19;
        bb7[20] =    i - 20;
        bb7[21] =    i - 21;
        bb7[22] =    i - 22;
        bb7[23] =    i - 23;
        bb7[24] =    i - 24;
        bb7[25] =    i - 25;
        bb7[26] =    i - 26;
        bb7[27] =    i - 27;
        bb7[28] =    i - 28;
        bb7[29] =    i - 29;
        bb7[30] =    i - 30;
        bb7[31] =    i - 31;
        bb7[32] =    i - 32;
        bb7[33] =    i - 33;
        bb7[34] =    i - 34;
        bb7[35] =    i - 35;
        bb7[36] =    i - 36;
        bb7[37] =    i - 37;
        bb7[38] =    i - 38;
        bb7[39] =    i - 39;
    }
    return 0;
}
