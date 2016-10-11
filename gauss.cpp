#include "gauss.h"

using namespace std;

Gauss::Gauss(int s/* = 1*/, int n_threads/* = 1*/) :size(s), num_threads(n_threads), elapsed_time(0.0)
{
	omp_set_num_threads(num_threads);
	A = new double[size*size];
	b = new double[size];
	x = new double[size];
	srand(unsigned int(time(NULL)));
	/*将初始矩阵设置为上三角阵*/
	for (int i = 0; i < size; i++)
	{
		b[i] = double(1 + rand() % 10);
		for (int j = 0; j < size; j++)
		{
			if (i > j)
				A[i*size + j] = 0; 
			else
				A[i*size + j] = double(1 + rand() % 10);
		}
			
	}
}

void Gauss::RowSerial()
{
	double sum = 0;
	for (int row = size - 1; row >= 0; row--)
	{
		sum = b[row];
		for (int col = row + 1; col < size; col++)
			sum -= A[row*size + col] * x[col];
		x[row] = sum / A[row*size + row];
	}
}

void Gauss::ColSerial()
{
	for (int row = 0; row < size; row++)
		x[row] = b[row];
	for (int col = size - 1; col >= 0; col--)
	{
		x[col] /= A[col*size + col];
		for (int row = 0; row < col; row++)
			x[row] -= A[row*size + col] * x[col];
	}
}

void Gauss::RowParallel()
{
	double sum = 0;
	
	#pragma omp parallel shared(sum)
	for (int row = size - 1; row >= 0; row--)
	{
		#pragma omp single
		sum = b[row];
		#pragma omp for reduction(+:sum)
		for (int col = row + 1; col < size; col++)
			sum -= A[row*size + col] * x[col];
		#pragma omp single
		x[row] = sum / A[row*size + row];
	}
}

void Gauss::ColParallel()
{
	#pragma omp parallel default(none)
	{
		#pragma omp for
		for (int row = 0; row < size; row++)
			x[row] = b[row];
		for (int col = size - 1; col >= 0; col--)
		{
			#pragma omp single
			x[col] /= A[col*size + col];
			#pragma omp for
			for (int row = 0; row < col; row++)
				x[row] -= A[row*size + col] * x[col];
		}
	}
}

void Gauss::RowParallelSchedule()
{
	int num_per_thread = size / omp_get_num_threads();//每个线程处理的迭代数
	double sum = 0;
	#pragma omp parallel
	for (int row = size - 1; row >= 0; row--)
	{
		#pragma omp single
		sum = b[row];
		#pragma omp for reduction(+:sum) schedule(dynamic)
		for (int col = row + 1; col < size; col++)
			sum -= A[row*size + col] * x[col];
		#pragma omp single
		x[row] = sum / A[row*size + row];
	}
}

void Gauss::ColParallelSchedule()
{
	int num_per_thread = size / omp_get_num_threads();//每个线程处理的迭代数
	#pragma omp parallel default(none)
	for (int row = size - 1; row >= 0; row--)
	{
		#pragma omp for
		for (row = 0; row < size; row++)
			x[row] = b[row];
		for (int col = size - 1; col >= 0; col--)
		{
			#pragma omp single
			x[col] /= A[col*size + col];
			#pragma omp for schedule(dynamic)
			for (row = 0; row < col; row++)
				x[row] -= A[row*size + col] * x[col];
		}
	}
}

void Gauss::Print()
{
	/*cout << "A:" << endl;
	for (int i = 0; i < size*size; i++)
	{
		cout << setiosflags(ios_base::left)
			<< "A[" << i << "] = "
			<< setw(20) << A[i];
		if (0 == (i + 1) % 5)
			cout << endl;
	}

	cout << "b:" << endl;
	for (int i = 0; i < size; i++)
	{
		cout << setiosflags(ios_base::left)
			<< "b[" << i << "] = "
			<< setw(20) << b[i];
		if (0 == (i + 1) % 5)
			cout << endl;
	}
*/
	cout << "x:" << endl;
	for (int i = 0; i < size; i++)
	{
		cout << setiosflags(ios_base::left)
			<< "x[" << i << "] = "
			<< setw(20) << x[i];
		if (0 == (i + 1) % 5)
			cout << endl;
	}
}

void Gauss::GetTime(Mode mode)
{
	double t_start = omp_get_wtime();
	double t_end;
	currentmode = mode;
	switch (currentmode)
	{
	case RowMajorSerial:
		RowSerial();
		t_end = omp_get_wtime();
		elapsed_time = t_end - t_start;
		cout << "行优先串行运行时间为：" << elapsed_time << "s" << endl;
		break;
	case ColMajorSerial:
		ColSerial();
		t_end = omp_get_wtime();
		elapsed_time = t_end - t_start;
		cout << "行优先串行运行时间为：" << elapsed_time << "s" << endl;
		break;
	case RowMajorParallel:	
		RowParallel();
		t_end = omp_get_wtime();
		elapsed_time = t_end - t_start;
		cout << "行优先并行运行时间为：" << elapsed_time << "s" << endl;
		break;
	case ColMajorParallel:
		ColParallel();
		t_end = omp_get_wtime();
		elapsed_time = t_end - t_start;
		cout << "列优先并行运行时间为：" << elapsed_time << "s" << endl;
		break;
	case RowMajorSchedule:
		RowParallelSchedule();
		t_end = omp_get_wtime();
		elapsed_time = t_end - t_start;
		cout << "行优先(schedule)并行运行时间为：" << elapsed_time << "s" << endl;
		break;
	case ColMajorSchedule:
		ColParallelSchedule();
		t_end = omp_get_wtime();
		elapsed_time = t_end - t_start;
		cout << "列优先(schedule)串行运行时间为：" << elapsed_time << "s" << endl;
		break;
	default:
		break;
	}
}

Gauss::~Gauss()
{
	delete[] A;
	delete[] b;
	delete[] x;
}