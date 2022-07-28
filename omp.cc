#include<iostream>
#include<cmath>
#include<omp.h>
#include<stdio.h>
#include<stdlib.h>
#include<sys/time.h>

using namespace std;


#define NUM_THREADS 7

const int MAXN=2048;
int N;
const double eps=1e-3;

float a[MAXN][MAXN];
float b[MAXN][MAXN];

char cases[6][20]={"input64.dat","input128.dat","input256.dat","input512.dat","input1024.dat","input2048.dat"};
int range[6]={64,128,256,512,1024,2048};

class CTimer
{
public:
	CTimer(void);
	~CTimer(void);

	void time_in();
	long long time_out();

private:
	struct timeval start;
	struct timeval end;
};

CTimer::CTimer(void)
{
}


CTimer::~CTimer(void)
{
}

void CTimer::time_in()
{
	gettimeofday(&start,NULL);
}

long long CTimer::time_out()
{
	gettimeofday(&end,NULL);
	long long timer=1000000 * (end.tv_sec-start.tv_sec)+ end.tv_usec-start.tv_usec;
	return timer;
}

void read(){
	for(int i=0;i<N;i++)
		for(int j=0;j<N;j++){
			cin>>a[i][j];
		}
	for(int i=0;i<N;i++)
		for(int j=0;j<N;j++){
			cin>>b[i][j];
		}
}


class Trivial{
public:
	float A[MAXN][MAXN];
	float res[MAXN][MAXN];
	void read(){
		for(int i=0;i<N;i++)
			for(int j=0;j<N;j++){
				A[i][j]=a[i][j];
			}
		for(int i=0;i<N;i++)
			for(int j=0;j<N;j++){
				res[i][j]=b[i][j];
			}
	}
	void calculate(){
		CTimer ct;
		ct.time_in();
		for(int k=0;k<N;k++){
			for(int j=k+1;j<N;j++){
				A[k][j]=A[k][j]/A[k][k];
			}
			A[k][k]=1.0f;
			for(int i=k+1;i<N;i++){
				for(int j=k+1;j<N;j++){
					A[i][j]=A[i][j]-A[i][k]*A[k][j];
				}
				A[i][k]=0;
			}
		}
		printf("trivial algo costs:%lldus",ct.time_out());
	}
	void check(){
		bool flag=0;
		for(int i=0;i<N;i++)
			for(int j=0;j<N;j++){
				if(abs(A[i][j]-res[i][j])>eps)flag=1;
			}
		if(flag){
			cout<<"wrong"<<endl;
		}
		else{
			cout<<"correct"<<endl;
		}
	}

}trivial_algo;


class OMP{
public:
	float res[MAXN][MAXN];
	void read(){

		for(int i=0;i<N;i++)
			for(int j=0;j<N;j++){
				res[i][j]=b[i][j];
			}
	}

	void calculate(){
		CTimer ct;
		ct.time_in();
		int i,j,k,tmp;
		#pragma omp parallel num_threads(NUM_THREADS) private(i,j,k,tmp)
		for(k = 0; k < N; ++k){
			#pragma omp single
			{
				float32x4_t vl= vld1q_dup_f32(&a[k][k]);
				for(int j=k+1;j+4<=N;j+=4){
					float32x4_t va = vld1q_f32(&a[k][j]);
					va=vdivq_f32(va,vl);
					vst1q_f32(&a[k][j],va);
				}
				a[k][k]=1.0f;
			}
			#pragma omp for num_threads(NUM_THREADS)
			for(i = k + 1; i < N; ++i){
				float32x4_t vaik= vld1q_dup_f32(&a[i][k]);
				for(int j=k+1;j+4<=N;j+=4){
					float32x4_t vakj = vld1q_f32(&a[k][j]);
					float32x4_t vaij = vld1q_f32(&a[i][j]);
					float32x4_t vx=vmulq_f32(vakj,vaik);
					vaij=vsubq_f32(vaij,vx);
					vst1q_f32(&a[i][j],vaij);
				}
				a[i][k]=0;
			}
		}
		printf("OMP algo costs:%lldus",ct.time_out());
	}
	void check(){
		bool flag=0;
		for(int i=0;i<N;i++)
			for(int j=0;j<N;j++){
				if(abs(a[i][j]-res[i][j])>eps){if(flag==0)cout<<a[i][j]<<" "<<res[i][j];flag=1;}
			}
		if(flag){
			cout<<"wrong"<<endl;
		}
		else{
			cout<<"correct"<<endl;
		}
	}
}OMP_algo;



int main() {
	for(int i=0;i<6;i++){
		N=range[i];
		freopen(cases[i],"r",stdin);
		cout<<cases[i]<<" condition:"<<endl;
		read();
		trivial_algo.read();
		trivial_algo.calculate();
		trivial_algo.check();
		OMP_algo.read();
		OMP_algo.calculate();
		OMP_algo.check();
	}
	return 0;
}
