#include "gauss.h"

int main(int argc, char **argv)
{
	int size = 100;//�����ģ
	int num_threads = 4;//�߳���
	Gauss gauss(size, num_threads);
	gauss.GetTime(RowMajorSerial);
	gauss.Print();
	gauss.GetTime(ColMajorSerial);
	gauss.Print();
	gauss.GetTime(RowMajorParallel);
	gauss.Print();
	gauss.GetTime(ColMajorParallel);
	gauss.Print();
	gauss.GetTime(RowMajorSchedule);
	gauss.Print();
	gauss.GetTime(ColMajorSchedule);
	gauss.Print();

	return 0;
}