#include <array>
#include <string>
#include <mpi.h>

//////////////////////////////////////////////////////////////////////////////////////////////////
// mpiexec options
// -aa (affinitiy auto: processes on the same node are assigned to different cores

//////////////////////////////////////////////////////////////////////////////////////////////////
// 15-Puzzle

// PMoveT describes the movement of the blank tile (has to include a special code for: no move)
enum PMoveT : uint8_t { NoMove = 0, Left = 1, Up = 2, Right = 3, Down = 4 };

#include "Master.hpp"

//////////////////////////////////////////////////////////////////////////////////////////////////
// A state is encoded by a linearization of the NTiles tiles.
// Tile number is stored with NBitsPerTile
// first tile is stored in most significant bits
struct PStateT {
	uint64_t m_board;	// packed board code
	uint8_t m_blank;	// zero based index of blank position

	PStateT(uint64_t board = 0, uint8_t blank = 0) : m_board(board), m_blank(blank) {}
	PStateT(const int arr[]);

	PStateT operator=(uint64_t board)			{ m_board = board; return *this; }
	bool operator==(const PStateT& st) const	{ return m_board == st.m_board; }
	bool isValid() const						{ return m_board != 0; }
	void toArray(int arr[]) const;

	void invalidate()							{ m_board = 0; }
};
namespace std {
	template<> struct hash<PStateT> {
		uint64_t operator()(const PStateT& st) const {
			return st.m_board;
		}
	};
}
typedef Node<PStateT, PMoveT> PNode;
typedef Master<PStateT, PMoveT> PMaster;

//////////////////////////////////////////////////////////////////////////////////////////////////
// Constants
const PathLenT MaxMoves = 50;						// maximum number of moves
//const int MaxSynchStates = 10;					// maximum number of synchronized states, must be increased to reduce communication overhead
const int NBitsPerTile = 4;							// number of bits per tile
const int NTilesPerRow = 4;							// number of tiles in one row or column
const int NTiles = NTilesPerRow*NTilesPerRow;		// number of tiles including black (= 0)
const uint64_t TileMask = (1 << NBitsPerTile) - 1;	// mask for one tile
const int GoalArray[NTiles] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0 };
const PStateT Goal(GoalArray);						// goal state

static_assert(1 << NBitsPerTile >= NTiles, "NBitsPerTile is too small");
static_assert(NBitsPerTile <= 8, "NBitsPerTile is too great");

//////////////////////////////////////////////////////////////////////////////////////////////////
// Static members
template<>
const PMoveT PNode::s_FirstMove = Left;										// range of allowed moves [FirstMove, LastMove]
template<>
const PMoveT PNode::s_LastMove = Down;										// "
template<>
const PMoveT PNode::s_ReverseMoves[] = { NoMove, Right, Down, Left, Up };	// reverse moves

//////////////////////////////////////////////////////////////////////////////////////////////////
void PStateT::toArray(int arr[]) const {
	PStateT st(*this);

	for (int i = NTiles - 1; i >= 0; i--) {
		arr[i] = st.m_board & TileMask;
		st.m_board >>= NBitsPerTile;
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////
PStateT::PStateT(const int arr[]) : m_board(0), m_blank(255) {
	for (int i = 0; i < NTiles; i++) {
		m_board = (m_board << NBitsPerTile) | (TileMask & arr[i]);
		if (arr[i] == 0) m_blank = i;
	}
	assert(m_blank >= 0 && m_blank < NTiles);
}

///////////////////////////////////////////////////////////////////////////
ostream& operator<<(ostream& os, const PStateT& s) {
	uint64_t board = s.m_board;
	uint8_t bitPos = (NTiles - 1)*NBitsPerTile;

	os << '{' << endl << (board >> bitPos);
	bitPos -= NBitsPerTile;
	for (int i = 1; i < NTiles; i++) {
		os << ',';
		if (i % NTilesPerRow == 0) os << endl;
		os << ((board >> bitPos) & TileMask);

		bitPos -= NBitsPerTile;
	}
	os << endl << '}';

	return os;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
// Check if move m is allowed in current state.
template<>
bool PNode::isPossible(PMoveT m) const {
	switch (m) {
	case Left:
		// only possible if blank is not in the left most column
		return m_state.m_blank % NTilesPerRow > 0;
	case Up:
		// only possible if blank is not in the top most row
		return m_state.m_blank / NTilesPerRow > 0;
	case Right:
		// only possible if blank is not in the right most column
		return m_state.m_blank % NTilesPerRow < NTilesPerRow - 1;
	case Down:
		// only possible if blank is not in the bottom most row
		return m_state.m_blank / NTilesPerRow < NTilesPerRow - 1;
	default:
		assert(false);
		return false;
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////
static void swapBits(uint64_t& board, uint8_t oldPos, uint8_t newPos) {
	uint64_t maskO = TileMask << (NTiles - 1 - oldPos)*NBitsPerTile;
	uint64_t maskN = TileMask << (NTiles - 1 - newPos)*NBitsPerTile;
	assert((board & maskO) == 0);
	if (oldPos < newPos) {
		uint8_t diff = newPos - oldPos;
		board |= ((board & maskN) << diff*NBitsPerTile);
	} else {
		uint8_t diff = oldPos - newPos;
		board |= ((board & maskN) >> diff*NBitsPerTile);
	}
	board &= ~maskN;
	assert((board & maskN) == 0);
}

//////////////////////////////////////////////////////////////////////////////////////////////////
// Apply valid move m.
template<>
void PNode::apply(PMoveT m) {
	assert(isPossible(m));

	uint8_t oldBlankPos = m_state.m_blank;

	switch (m) {
	case Left:
		m_state.m_blank--;
		break;
	case Up:
		m_state.m_blank -= NTilesPerRow;
		break;
	case Right:
		m_state.m_blank++;
		break;
	case Down:
		m_state.m_blank += NTilesPerRow;
		break;
	default:
		assert(false);
	}

	swapBits(m_state.m_board, oldBlankPos, m_state.m_blank);
	m_move = m;
	m_nMoves++;
}

///////////////////////////////////////////////////////////////////////////
// Admissible heuristic function for path length between this and target. 
// manhatten distance to target of all tiles, but blank
// goal state: tile position = (tile number - 1) mod NTiles
template<>
PathLenT PNode::heuristic(const PStateT target) const {
	// TODO
	return 0;
}

///////////////////////////////////////////////////////////////////////////
// User defined
// Describe type PStateT for MPI. len is the array length of types and blockLengths. 
// types are the corresponding MPI types of the attributes.
// blockLengths is used in case of arrays and specifies the number of elements per attribute.
// offsets are the offsets of the attributes
// If types is nullptr then you have to return in len the number of attributes in PStateT.
template<>
void PNode::getMPIstateDescription(MPI_Datatype types[], int blockLengths[], MPI_Aint offsets[], int& len) {
	if (types) {
		assert(len >= 2);
		types[0] = MPI_UINT64_T;	// corresponds to PStateT.m_board
		types[1] = MPI_UINT8_T;		// corresponds to PStateT.m_blank
		blockLengths[0] = 1;
		blockLengths[1] = 1;
		offsets[0] = offsetof(PStateT, m_board);
		offsets[1] = offsetof(PStateT, m_blank);
	} else {
		len = 2;
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////
void puzzleTest(int maxSynchStates) {
	int nProcs, myID;

	MPI_Comm_rank(MPI_COMM_WORLD, &myID);
	MPI_Comm_size(MPI_COMM_WORLD, &nProcs);

	if (nProcs >= 3 && (nProcs & 1) == 1) {
		//array<int, NTiles> arr = { 2,5,1,3, 10,7,8,4, 6,13,0,11, 9,14,15,12 };					// solvable in 24 moves
		//array<int, NTiles> arr = { 1,2,3,8, 6,0,7,11, 5,10,14,4, 9,13,12,15 }; 					// solvable in 28 moves
		//array<int, NTiles> arr = { 1,3,6,4, 2,14,12,10, 9,7,0,11, 5,13,15,8 };					// solvable in 32 moves
		//array<int, NTiles> arr = { 5,1,7,6, 2,4,3,8, 14,9,12,15, 13,10,11,0 };					// solvable in 34 moves
		//array<int, NTiles> arr = { 2,11,3,6, 1,9,10,8, 13,5,0,4, 14,12,7,15 };					// solvable in 36 moves
		//array<int, NTiles> arr = { 2,7,0,10, 1,13,4,3, 12,14,6,8, 5,11,9,15 };					// solvable in 38 moves
		array<int, NTiles> arr = { 3,1,4,7, 2,11,6,12, 0,15,13,5, 9,10,14,8 };					// solvable in 40 moves
		//array<int, NTiles> arr = { 13,9,5,2, 11,1,3,0, 15,6,7,8, 12,14,4,10 };					// solvable in 46 moves

		if (myID == 0) {
			cout << "Puzzle with " << nProcs << " MPI processes" << endl;

			//int ignore; cin >> ignore;	// just for debugging

			// create hard instances with given solution: comment the following lines if you want to use the provided text example
			//PNode node = PNode::createInstance(Goal, "1112334143432114122233344141122233344411232344");	// 46 moves, 1075 s
			//PNode node = PNode::createInstance(Goal, "3334112332112143414333211432232111433344");		// 40 moves, 294 s (4 processes)
			//node.getState().toArray(arr.data());

			// generate random instance with MaxMoves moves: comment the following two lines if you want to use the provided text example
			//PNode node = PNode::randomInstance(Goal, MaxMoves);
			//node.getState().toArray(arr.data());
			//cout << node << endl;

			//PNode gn(Goal); assert(gn.heuristic(Goal) == 0);
		}

		// send arr to all processes 
		MPI_Bcast(arr.data(), NTiles, MPI_INT, 0, MPI_COMM_WORLD);

		// use barrier to synchronize start time
		MPI_Barrier(MPI_COMM_WORLD);
		double startTime = MPI_Wtime();

		const PStateT start(arr.data());
		//cout << start << endl;

		// search solution
		string result = PMaster::search(myID, nProcs, MaxMoves, maxSynchStates, start, Goal);

		// stop time
		double localElapsed = MPI_Wtime() - startTime, elapsed;

		// reduce maximum time
		MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

		if (myID == 0) {
			// check solution
			if (!result.empty() && result[0] < 'A') {
				cout << "Path [" << result << "] (" << result.size() << " moves) is ";
				cout << (PMaster::checkSolution(start, Goal, result) ? "valid!" : "invalid!") << endl;
				cout << elapsed << " s" << endl;

				if (result.size() > MaxMoves) {
					// something is wrong
					cout << start << endl;
				}
			}

			// show result
			if (!result.empty() && result[0] >= 'A') {
				cout << result << endl;

				if (result[0] == 'N') {
					cout << start << endl;
				}
			}
		}
	} else {
		if (myID == 0) {
			cerr << "Number of processes (" << nProcs << ") must be at least 3 and odd" << endl;
		}
	}
}
