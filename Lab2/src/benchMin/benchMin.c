// speedup = 1.000000

#include <stdlib.h>

#define N 381

int main() {

	register int i, j;
        int count = 0;
	int x, y, z, w;
	short a[N] = {0}, b;
	register short c, d;

	for (i = 0; i < N; i++){
		x = rand() % N;
		y = rand() % N;
		z = rand() % N;
		w = rand() % N;

		if (x == 0) // some random jumps
			goto case_c;
		else if (x == 1)
			goto case_b;
		else
			goto no_case_yet; // always jumps. on first iteration there will be a miss
		goto case_a; // these three lines are never run. On the branch miss of the previous line, these instructions are issued, the goto may even be executed, but are lost afterwards
		j = rand()%N;
		c = a[j];
no_case_yet:
		if (x < N / x) // random compared to sqrt(N); the division will also stall the branch for longer, causing more missed instructions
			goto case_a;
		else if (y < N / y) // a lot of possible cases, to minimize the branch hit rates
			goto case_b;
		else if (z < N / z) // gotos for each case, to ensure that the not taken approach will always fail, including the first iteration
			goto case_c;
		else if (w < N / w)
			goto case_d;
		else
			goto case_e;

cases:
	}
return 0;
	case_a:
		count++;
		j = rand() % N;
		if (j > N / j)
		goto case_b;
		else
		goto cases;

	case_b:
		count--;
		j = rand() % N;
		if (j > N / j)
		goto case_c;
		else
		goto cases;

	case_c:
		count = i;
		j = rand() % N;
		if (j > N / j)
		goto case_d;
		else
		goto cases;

	case_d:
		count = 0;
		j = rand() % N;
		if (j > N / j)
		goto case_e;
		else
		goto cases;

	case_e:
		count = 1;
		j = rand() % N;
		if (j > N / j)
		goto case_a;
		else
		goto cases;

	return count;
}

