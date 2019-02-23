#pragma once

#include <queue>
#include "Node.h"
#include "OpenList.h"
#include "ClosedList.h"

//////////////////////////////////////////////////////////////////////////////////////////////////
// Searcher: a process that expands an open node (state), creates new open nodes and adds
// the expanded node to the closed nodes.
template<class StateT, class MoveT>
class Searcher {
	OpenList<StateT, MoveT> m_open;			// open nodes (states), there are no states which are in open and closed list at the same time
	ClosedList<StateT, MoveT> m_closed;		// seen and handled nodes (states), can also be foreign nodes
	deque<StateT> m_synchStates;			// states selected for synchronization with other processes
	Node<StateT, MoveT> *m_workBuf;			// buffer used during exchanging work
	Node<StateT, MoveT> *m_synchBuf;		// synchronization buffer
	int m_myID;								// my process rank
	int m_nProcs;							// number of processes
	PathLenT m_maxMoves;					// maximum number of moves
	StateT m_start, m_goal;					// start and goal states (the same for all processes)
	MPI_Comm m_group;						// processes with the same root, group size = nProcs/2
	MPI_Datatype m_stateType;				// MPI data type for state exchange
	MPI_Request *m_workRequests;			// pending work requests

	const int m_MaxSynchStates;				// maximum number of states to be sychnronized

public:
	///////////////////////////////////////////////////////////////////////////
	Searcher(int myID, int nProcs, PathLenT maxMoves, int maxSynchStates, const StateT& start, const StateT& goal)
		: m_myID(myID)
		, m_nProcs(nProcs)
		, m_maxMoves(maxMoves)
		, m_workBuf(new Node<StateT, MoveT>[maxSynchStates])		// used only in group communication
		, m_synchBuf(new Node<StateT, MoveT>[maxSynchStates])		// used only in synchronization
		, m_start(start)
		, m_goal(goal)
		, m_stateType(Node<StateT, MoveT>::createMPItype())
		, m_MaxSynchStates(maxSynchStates)
		, m_workRequests(new MPI_Request[nProcs/2])
	{
		assert(m_maxMoves < NoSolution);

		// create process group
		MPI_Comm_split(MPI_COMM_WORLD, 1 + rootIsGoal(m_myID), (m_myID - 1)/2, &m_group);
		//int nGroupSize; MPI_Comm_size(m_group, &nGroupSize); assert(nGroupSize == nProcs/2);
	}

	~Searcher() {
		delete[] m_synchBuf;
		delete[] m_workBuf;
		delete[] m_workRequests;
		MPI_Comm_free(&m_group);
		MPI_Type_free(&m_stateType);
	}

	void addOpenState(const StateT& sr) { Node<StateT, MoveT> *node = new Node<StateT, MoveT>(sr); m_open.add(node); }
	void search(const StateT& target);

	///////////////////////////////////////////////////////////////////////////
	static bool rootIsGoal(int id) { return id & 1; }

private:
	bool addNewNode(Node<StateT, MoveT> *node);
	void communication(PathLenT& maxSearchCost, MPI_Request& synchRequest);
};
