#include <stdlib.h>

#define N 40923

int main(){
	
	register int i, count = 0;
	int x;

	for (i = 0; i < N; i++){
		x = rand()%N;
		if (x < N/2)
			goto case_a;
		else
			goto case_b;
		cases:
	}
	return count;	

case_a:
	count++;
	goto cases;
case_b:
	count = count;
	goto cases;
}
