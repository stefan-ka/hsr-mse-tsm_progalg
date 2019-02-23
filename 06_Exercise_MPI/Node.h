#pragma once

#include <iostream>
#include <cassert>
#include <cstdint>
#include <type_traits>
#include <mpi.h>

using namespace std;

//////////////////////////////////////////////////////////////////////////////////////////////////
// Types
typedef uint8_t PathLenT;

//////////////////////////////////////////////////////////////////////////////////////////////////
// Constants
const MPI_Datatype MPI_PathLen_Type = MPI_UINT8_T;
const MPI_Datatype MPI_Move_Type = MPI_UINT8_T;
const int InvalidFlag = -1;					// used in s_rootIsGoal

//////////////////////////////////////////////////////////////////////////////////////////////////
// A node (state) in search space.
template<class StateT, class MoveT>
class Node {
	StateT m_state;			// state
	PathLenT m_nMoves;		// number of moves
	MoveT m_move;			// last move done before this state, in root nodes: NoMove

	static_assert(is_same<PathLenT, uint8_t>::value, "PathLenT is not uint8_t");
	static_assert(sizeof(MoveT) <= sizeof(uint8_t), "MoveT is not storage compatible to uint8_t");

public:
	static const MoveT s_ReverseMoves[];	// NoMove and moves in reverse direction
	static const MoveT s_FirstMove;			// range of allowed moves
	static const MoveT s_LastMove;			// range of allowed moves

	static StateT s_start, s_goal;			// start and goal states (the same for all processes)
	static int s_rootIsGoal;				// 1: root node is goal state; 0: root node is start state; -1: invalid usage 

	///////////////////////////////////////////////////////////////////////////
	Node()
		: m_nMoves(0)
		, m_move(NoMove)
	{}
	Node(const StateT& s)
		: m_state(s)
		, m_nMoves(0)
		, m_move(NoMove)
	{}
	Node(Node* node, MoveT m)
		: m_state(node->m_state)
		, m_nMoves(node->m_nMoves)
		, m_move(NoMove)
	{
		assert(node);
		apply(m);
	}

	void invalidate()						{ m_state.invalidate(); m_nMoves = 0; }
	void setNumMoves(PathLenT n)			{ m_nMoves = n; }

	bool isRoot() const						{ return m_move == NoMove; }
	bool isTarget(bool rootIsGoal) const	{ return m_state == ((rootIsGoal) ? s_start : s_goal); }
	bool hasState(const StateT& st) const	{ return m_state == st; }
	bool reversesLastMove(MoveT m) const	{ return m == s_ReverseMoves[m_move]; }
	StateT getState() const					{ return m_state; }
	MoveT getMove() const					{ return m_move; }
	PathLenT getNumMoves() const			{ return m_nMoves; }
	bool isValid() const					{ return m_state.isValid(); }
	StateT getTarget(bool rootIsGoal) const { return (rootIsGoal) ? s_start : s_goal; }
	PathLenT getCost(bool rootIsGoal) const { return heuristic(getTarget(rootIsGoal)) + m_nMoves; }
	PathLenT getHeuristic(bool rIsG) const	{ return heuristic(getTarget(rIsG)); }

	///////////////////////////////////////////////////////////////////////////
	// these methods must not be used in Master
	StateT getTarget() const				{ assert(s_rootIsGoal != InvalidFlag); return (s_rootIsGoal) ? s_start : s_goal; }
	PathLenT getCost() const				{ assert(s_rootIsGoal != InvalidFlag); return heuristic(getTarget()) + m_nMoves; }
	PathLenT getHeuristic() const			{ assert(s_rootIsGoal != InvalidFlag); return heuristic(getTarget()); }
	bool operator<(const Node& v) const		{ assert(s_rootIsGoal != InvalidFlag); return getCost() <  v.getCost(); }

	///////////////////////////////////////////////////////////////////////////
	// write node code to stream os and return os
	friend ostream& operator<<(ostream& os, const Node& node) { return os << node.m_state; }

	//////////////////////////////////////////////////////////////////////////////////////////////////
	// reverse move
	static MoveT reverse(MoveT m) { assert(s_FirstMove <= m && m <= s_LastMove);  return s_ReverseMoves[m]; }

	//////////////////////////////////////////////////////////////////////////////////////////////////
	// create MPI data type for struct Node
	static MPI_Datatype createMPItype() {
		MPI_Datatype stateType;
		int len = 0;

		getMPIstateDescription(nullptr, nullptr, nullptr, len);
		len += 2; // Node has 2 additional attributes
		int *blockLengths = new int[len];
		MPI_Datatype *types = new MPI_Datatype[len];
		MPI_Aint *offsets = new MPI_Aint[len];

		getMPIstateDescription(types, blockLengths, offsets, len);

		types[len - 2] = MPI_PathLen_Type;
		blockLengths[len - 2] = 1;
		offsets[len - 2] = offsetof(Node, m_nMoves);

		types[len - 1] = MPI_Move_Type;
		blockLengths[len - 1] = 1;
		offsets[len - 1] = offsetof(Node, m_move);

		MPI_Type_create_struct(len, blockLengths, offsets, types, &stateType);
		MPI_Type_commit(&stateType);

		//int s; MPI_Type_size(m_stateType, &s);	// number of bytes: 10

		delete[] blockLengths;
		delete[] types;
		delete[] offsets;

		return stateType;
	}

	//////////////////////////////////////////////////////////////////////////////////////////////////
	// Generates a valid sequence of n randomly chosen moves.
	// Returns the start state.
	static Node randomInstance(const StateT& goal, int n) {
		Node node(goal);
		MoveT lastMove;
		
		do {
			lastMove = (MoveT)(s_FirstMove + rand()*s_LastMove/RAND_MAX);
		} while (!node.isPossible(lastMove));

		//cout << (int)lastMove;
		node.apply(lastMove);

		for (int i = 1; i < n; i++) {
			// next move
			MoveT move;
			
			do {
				move = (MoveT)(s_FirstMove + rand()*(s_LastMove - 1)/RAND_MAX);

				if (move == s_ReverseMoves[lastMove]) {
					// it's the reverse move -> replace it
					move = s_LastMove;
				}
			} while (!node.isPossible(move));
			
			//cout << (int)move;
			node.apply(move);
			lastMove = move;
		}
		//cout << endl;
		return node;
	}

	//////////////////////////////////////////////////////////////////////////////////////////////////
	// Generates a problem instance with the given solution.
	// Returns the start state.
	static Node createInstance(const StateT& goal, const string& solution) {
		Node node(goal);

		for (int i = (int)solution.size() - 1; i >= 0; i--) {
			// next move
			MoveT move = reverse((MoveT)(solution[i] - '0'));

			//cout << (int)move;
			node.apply(move);
		}
		//cout << endl;
		return node;
	}

	///////////////////////////////////////////////////////////////////////////
	// User defined
	// Return true if move m is possible in the current state
	bool isPossible(MoveT m) const;

	///////////////////////////////////////////////////////////////////////////
	// User defined
	// Apply valid move m.
	void apply(MoveT m);

	///////////////////////////////////////////////////////////////////////////
	// User defined
	// Addmissible heuristic function for path length between this and target. 
	// Can always return 0.
	PathLenT heuristic(const StateT target) const;

	///////////////////////////////////////////////////////////////////////////
	// User defined
	// Describe type StateT for MPI. len has to be the number of attributes in type StateT. 
	// types are the corresponding MPI types of the attributes.
	// blockLengths is used in case of arrays and specifies the number of elements per attribute.
	// offsets are the offsets of the attributes
	// if types is nullptr then it returns only len
	static void getMPIstateDescription(MPI_Datatype types[], int blockLengths[], MPI_Aint offsets[], int& len);
};

