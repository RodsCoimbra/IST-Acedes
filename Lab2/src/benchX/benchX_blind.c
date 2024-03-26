#define N 40

int main()
{
    int A[N][N];
    int B[N][N];
    int C[N][N] = {};

    register int i, j, k, *aa, *bb, *cc, temp, s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15;

    for (i = 0; i < N; i++)
    {
        aa = A[i];
        bb = B[i];
        aa[0] = i;
        bb[0] = i;
        aa[1] = i + 1;
        bb[1] = i - 1;
        aa[2] = i + 2;
        bb[2] = i - 2;
        aa[3] = i + 3;
        bb[3] = i - 3;
        aa[4] = i + 4;
        bb[4] = i - 4;
        aa[5] = i + 5;
        bb[5] = i - 5;
        aa[6] = i + 6;
        bb[6] = i - 6;
        aa[7] = i + 7;
        bb[7] = i - 7;
        aa[8] = i + 8;
        bb[8] = i - 8;
        aa[9] = i + 9;
        bb[9] = i - 9;
        aa[10] = i + 10;
        bb[10] = i - 10;
        aa[11] = i + 11;
        bb[11] = i - 11;
        aa[12] = i + 12;
        bb[12] = i - 12;
        aa[13] = i + 13;
        bb[13] = i - 13;
        aa[14] = i + 14;
        bb[14] = i - 14;
        aa[15] = i + 15;
        bb[15] = i - 15;
        aa[16] = i + 16;
        bb[16] = i - 16;
        aa[17] = i + 17;
        bb[17] = i - 17;
        aa[18] = i + 18;
        bb[18] = i - 18;
        aa[19] = i + 19;
        bb[19] = i - 19;
        aa[20] = i + 20;
        bb[20] = i - 20;
        aa[21] = i + 21;
        bb[21] = i - 21;
        aa[22] = i + 22;
        bb[22] = i - 22;
        aa[23] = i + 23;
        bb[23] = i - 23;
        aa[24] = i + 24;
        bb[24] = i - 24;
        aa[25] = i + 25;
        bb[25] = i - 25;
        aa[26] = i + 26;
        bb[26] = i - 26;
        aa[27] = i + 27;
        bb[27] = i - 27;
        aa[28] = i + 28;
        bb[28] = i - 28;
        aa[29] = i + 29;
        bb[29] = i - 29;
        aa[30] = i + 30;
        bb[30] = i - 30;
        aa[31] = i + 31;
        bb[31] = i - 31;
        aa[32] = i + 32;
        bb[32] = i - 32;
        aa[33] = i + 33;
        bb[33] = i - 33;
        aa[34] = i + 34;
        bb[34] = i - 34;
        aa[35] = i + 35;
        bb[35] = i - 35;
        aa[36] = i + 36;
        bb[36] = i - 36;
        aa[37] = i + 37;
        bb[37] = i - 37;
        aa[38] = i + 38;
        bb[38] = i - 38;
        aa[39] = i + 39;
        bb[39] = i - 39;
    }

    for (i = 0; i < N; i++)
    {
	aa = A[i];
 	cc = C[i];
        for (k = 0; k < N; k++)
        {
            temp = aa[k];
            bb = B[k];
            s0 = temp * bb[0];
            s1 = temp * bb[1];
            s2 = temp * bb[2];
            s3 = temp * bb[3];
            s4 = temp * bb[4];
            s5 = temp * bb[5];
            s6 = temp * bb[6];
            s7 = temp * bb[7];
            s8 = temp * bb[8];
            s9 = temp * bb[9];
            s10 = temp * bb[10];
            s11 = temp * bb[11];
            s12 = temp * bb[12];
            s13 = temp * bb[13];
            s14 = temp * bb[14];
            s15 = temp * bb[15];
            cc[0] += s0;
            cc[1] += s1;
            cc[2] += s2;
            cc[3] += s3;
            cc[4] += s4;
            cc[5] += s5;
            cc[6] += s6;
            cc[7] += s7;
            cc[8] += s8;
            cc[9] += s9;
            cc[10] += s10;
            cc[11] += s11;
            cc[12] += s12;
            cc[13] += s13;
            cc[14] += s14;
            cc[15] += s15;
            s0 = temp * bb[16];
            s1 = temp * bb[17];
            s2 = temp * bb[18];
            s3 = temp * bb[19];
            s4 = temp * bb[20];
            s5 = temp * bb[21];
            s6 = temp * bb[22];
            s7 = temp * bb[23];
            s8 = temp * bb[24];
            s9 = temp * bb[25];
            s10 = temp * bb[26];
            s11 = temp * bb[27];
            s12 = temp * bb[28];
            s13 = temp * bb[29];
            s14 = temp * bb[30];
            s15 = temp * bb[31];
            cc[16] += s0;
            cc[17] += s1;
            cc[18] += s2;
            cc[19] += s3;
            cc[20] += s4;
            cc[21] += s5;
            cc[22] += s6;
            cc[23] += s7;
            cc[24] += s8;
            cc[25] += s9;
            cc[26] += s10;
            cc[27] += s11;
            cc[28] += s12;
            cc[29] += s13;
            cc[30] += s14;
            cc[31] += s15;
            s0 = temp * bb[32];
            s1 = temp * bb[33];
            s2 = temp * bb[34];
            s3 = temp * bb[35];
            s4 = temp * bb[36];
            s5 = temp * bb[37];
            s6 = temp * bb[38];
            s7 = temp * bb[39];
            cc[32] += s0;
            cc[33] += s1;
            cc[34] += s2;
            cc[35] += s3;
            cc[36] += s4;
            cc[37] += s5;
            cc[38] += s6;
            cc[39] += s7;
        }
    }
    return 0;
}

