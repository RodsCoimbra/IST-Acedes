#define N 40

int main(){
	int A[N][N];
	int B[N][N];
	int C[N][N] = {};

	register int i, j, k, *aa, *bb, *cc, temp, s0, s1, s2, s3, s4, s5, s6, s7;
	register int val1, val2;

	for (i = 0; i < N; i++) {
		aa = A[i];
		bb = B[i];
		cc = C[i];
		for (j = 0; j < N; j+=8) {
			aa[j] = i + j;
			aa[j+1] = i + j + 1;
			aa[j+2] = i + j + 2;
			aa[j+3] = i + j + 3;
			aa[j+4] = i + j + 4;
			aa[j+5] = i + j + 5;
			aa[j+6] = i + j + 6;
			aa[j+7] = i + j + 7;
			bb[j] = i - j;
			bb[j+1] = i - j - 1;
			bb[j+2] = i - j - 2;
			bb[j+3] = i - j - 3;
			bb[j+4] = i - j - 4;
			bb[j+5] = i - j - 5;
			bb[j+6] = i - j - 6;
			bb[j+7] = i - j - 7;
		}
	}

	for (i = 0; i < N; i++) {
		cc = C[i];
		aa = A[i];
		for (k = 0; k < N; k++) {
			temp = aa[k];
			bb = B[k];
			for (j = 0; j < N-8; j+=16) {
				s0 = temp * bb[j];
				s1 = temp * bb[j+1];
				s2 = temp * bb[j+2];
				s3 = temp * bb[j+3];
				s4 = temp * bb[j+4];
				s5 = temp * bb[j+5];
				s6 = temp * bb[j+6];
				s7 = temp * bb[j+7];
				cc[j] += s0;
				cc[j+1] += s1;
				cc[j+2] += s2;
				cc[j+3] += s3;
				cc[j+4] += s4;
				cc[j+5] += s5;
				cc[j+6] += s6;
				cc[j+7] += s7;	

				s0 = temp * bb[j+8];
				s1 = temp * bb[j+9];
				s2 = temp * bb[j+10];
				s3 = temp * bb[j+11];
				s4 = temp * bb[j+12];
				s5 = temp * bb[j+13];
				s6 = temp * bb[j+14];
				s7 = temp * bb[j+15];
				cc[j+8] += s0;
				cc[j+9] += s1;
				cc[j+10] += s2;
				cc[j+11] += s3;
				cc[j+12] += s4;
				cc[j+13] += s5;
				cc[j+14] += s6;
				cc[j+15] += s7;	

			}
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
