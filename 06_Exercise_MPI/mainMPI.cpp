#include <ctime>
#include <cstdlib>
#include <string>
#include <iostream>
#include <mpi.h>

using namespace std;

//////////////////////////////////////////////////////////////////////////////////////////////////
// Prototypes
void shellsort();
void puzzleTest(int maxSynchStates);

//////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[]) {
	srand((unsigned int)time(nullptr));

	if (argc == 2) {
		try {
			const int maxSynchStates = stoi(argv[1]);
			if (maxSynchStates <= 0) throw out_of_range("maxSynchStates has to be a positive integer value");

			MPI_Init(&argc, &argv);

			shellsort();
			puzzleTest(maxSynchStates);

			MPI_Finalize();
		}
		catch (logic_error& ex) {
			cerr << ex.what() << endl;
		}
	} else {
		cerr << "Usage: Exercise6_MPI maxSynchStates" << endl;
	}

	return 0;
}