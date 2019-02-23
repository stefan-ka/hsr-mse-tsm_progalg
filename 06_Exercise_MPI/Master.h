#pragma once
#include <queue>
#include "Node.h"
#include "ClosedList.h"

//////////////////////////////////////////////////////////////////////////////////////////////////
// Types
enum class MasterCommTags : int { Finish, LastStates, States, MaxSearchCost, NeedsWork, OfferWork };

//////////////////////////////////////////////////////////////////////////////////////////////////
// Constants
const bool VeryVerbose = false;				// true: additional console output
const bool Verbose = VeryVerbose || false;	// true: minimal console output
const PathLenT NoSolution = 255;			// no solution was found
const int NTimes = 4;						// counter + number of wall-clock times

//////////////////////////////////////////////////////////////////////////////////////////////////
template<class StateT, class MoveT>
class Master {
	ClosedList<StateT, MoveT> m_closed[2];	// seen and handled nodes (states) with root = Start [0] or Goal [1]
	Node<StateT, MoveT> *m_synchBuf;		// synchronization buffer
	int m_myID;								// my process rank
	int m_nProcs;							// number of processes
	PathLenT m_maxMoves;					// maximum number of moves
	StateT m_start, m_goal;					// start and goal states (the same for all processes)
	MPI_Comm m_group;						// processes with the same root, group size = nProcs/2
	MPI_Datatype m_stateType;				// MPI data type for state exchange

	const int m_MaxSynchStates;				// maximum number of states to be sychnronized

public:
	///////////////////////////////////////////////////////////////////////////
	Master(int myID, int nProcs, PathLenT maxMoves, int maxSynchStates, const StateT& start, const StateT& goal)
		: m_myID(myID)
		, m_nProcs(nProcs)
		, m_maxMoves(maxMoves)
		, m_synchBuf(new Node<StateT, MoveT>[maxSynchStates])		// used only in synchronization
		, m_start(start)
		, m_goal(goal)
		, m_stateType(Node<StateT, MoveT>::createMPItype())
		, m_MaxSynchStates(maxSynchStates)
	{
		assert(m_maxMoves < NoSolution);

		// create process group
		MPI_Comm_split(MPI_COMM_WORLD, 0, m_myID/2, &m_group);
		//int nGroupSize; MPI_Comm_size(m_group, &nGroupSize); assert(nGroupSize == 1);
	}

	~Master() {
		delete[] m_synchBuf;
		MPI_Type_free(&m_stateType);
	}

	///////////////////////////////////////////////////////////////////////////
	static string search(int myID, int nProcs, PathLenT maxMoves, const int maxSynchStates, const StateT& start, const StateT& goal);
	static bool rootIsGoal(int id) { return id & 1; }

	///////////////////////////////////////////////////////////////////////////
	static bool checkSolution(const StateT& start, const StateT& goal, const string& solution) {
		Node<StateT, MoveT> node(start);

		//cout << "Start   : " << node << endl;
		for (size_t i = 0; i < solution.size(); i++) {
			node.apply((MoveT)(solution[i] - '0'));
		}
		//cout << "Solution: " << node << endl;
		return node.hasState(goal);
	}

private:
	string searchSolution();
	void minimizePathLen(PathLenT& minPathLen, Node<StateT, MoveT>& minNodeG, Node<StateT, MoveT>& minNodeS, PathLenT pathLen, const Node<StateT, MoveT>& nodeG, const Node<StateT, MoveT>& nodeS);
	string buildPath(const Node<StateT, MoveT>& node, bool rootIsGoal, int pathLen);
	string buildPath(PathLenT pathLen, Node<StateT, MoveT>& minNodeG, Node<StateT, MoveT>& minNodeS);
};
