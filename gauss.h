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
	int size;//�����С
	int num_threads;//�߳���
	double elapsed_time;//�ķ�ʱ��
	double *A;
	double *b;
	double *x;
	Mode currentmode;
public:
	Gauss(int size = 1, int n_threads = 1);
	void RowSerial();//�����ȴ��м���
	void ColSerial();//�����ȴ��м���
	void RowParallel();//�����Ȳ��м���
	void ColParallel();//�����Ȳ��м���
	void RowParallelSchedule();//�����Ȳ��м��㣨��schedule����д��
	void ColParallelSchedule();//�����Ȳ��м��㣨��schedule����д��
	void Print();//������̽�
	void GetTime(Mode mode);//���ÿ�ּ��㷽������ʱ��
	~Gauss();
};
#endif // !GAUSS_H