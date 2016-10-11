#ifndef GAUSS_H
#define GAUSS_H

#include <iostream>
#include <omp.h>
#include <ctime>
#include <cstdio>
#include <iomanip>

enum Mode { RowMajorSerial, ColMajorSerial, RowMajorParallel, ColMajorParallel, RowMajorSchedule, ColMajorSchedule};

class Gauss
{
private:
	int size;//矩阵大小
	int num_threads;//线程数
	double elapsed_time;//耗费时间
	double *A;
	double *b;
	double *x;
	Mode currentmode;
public:
	Gauss(int size = 1, int n_threads = 1);
	void RowSerial();//行优先串行计算
	void ColSerial();//列优先串行计算
	void RowParallel();//行优先并行计算
	void ColParallel();//列优先并行计算
	void RowParallelSchedule();//行优先并行计算（用schedule语句改写）
	void ColParallelSchedule();//列优先并行计算（用schedule语句改写）
	void Print();//输出方程解
	void GetTime(Mode mode);//获得每种计算方法所费时间
	~Gauss();
};
#endif // !GAUSS_H