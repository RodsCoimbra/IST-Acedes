#define N 40

void matrix_multiply(int A[N][N], int B[N][N], int C[N][N]){
	
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			C[i][j] = 0;
			for (int k = 0; k < N; k++) {
				C[i][j] += A[i][k] * B[k][j];
			}
		}
	}
}

int main(){
	int A[N][N];
	int B[N][N];
	int C[N][N];

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			A[i][j] = i + j;
			B[i][j] = i - j;
			C[i][j] = 0;
		}
	}

	matrix_multiply(A, B, C);

	return 0;

}
