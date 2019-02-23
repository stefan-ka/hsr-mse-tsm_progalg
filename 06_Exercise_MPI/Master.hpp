#pragma once

#include <sstream>
#include <algorithm>
#include "Master.h"
#include "Searcher.hpp"

//////////////////////////////////////////////////////////////////////////////////////////////////
// Static members
template<class StateT, class MoveT> StateT Node<StateT, MoveT>::s_start;
template<class StateT, class MoveT> StateT Node<StateT, MoveT>::s_goal;
template<class StateT, class MoveT> int Node<StateT, MoveT>::s_rootIsGoal; 

//////////////////////////////////////////////////////////////////////////////////////////////////
// Processes with even myID will search towards goal, processes with odd myID will search from goal towards start
// Process with myID = 0 is the master and it doesn't process open states.
template<class StateT, class MoveT>
string Master<StateT, MoveT>::search(int myID, int nProcs, PathLenT maxMoves, const int maxSynchStates, const StateT& start, const StateT& goal) {
	Node<StateT, MoveT>::s_start = start;
	Node<StateT, MoveT>::s_goal = goal;
	Node<StateT, MoveT>::s_rootIsGoal = (myID) ? rootIsGoal(myID) : InvalidFlag;

	if (start == goal) {
		return "SOLUTION ALREADY FOUND";
	} else {
		string result;

		if (myID == 0) {
			// master process
			Master<StateT, MoveT> m(myID, nProcs, maxMoves, maxSynchStates, start, goal);

			result = m.searchSolution();
		} else {
			Searcher<StateT, MoveT> s(myID, nProcs, maxMoves, maxSynchStates, start, goal);

			if (myID == 1) {
				// searcher: search root is goal
				s.addOpenState(goal);
				s.search(start);
			} else if (myID == 2) {
				// searcher: search root is start
				s.addOpenState(start);
				s.search(goal);
			} else {
				// searchers: these processes start without any assigned search state
				s.search(rootIsGoal(myID) ? start : goal);
			}
		}
		return result;
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////
// Receive closed states of searchers and check for solutions.
template<class StateT, class MoveT>
string Master<StateT, MoveT>::searchSolution() {
	const int nSearchers = m_nProcs - 1;

	bool running = true;
	string result;
	MPI_Status status;
	PathLenT pathLen = NoSolution;
	PathLenT searchCost = 0, maxSearchCost = m_maxMoves;
	Node<StateT, MoveT> minNodeG, minNodeS;	// best nodes with root = Goal / Start
	int nLastStates = 0;
	double mTime = 0;

	//if (VeryVerbose) cout << "Master search started" << endl;

	do {
		// receive closed states
		MPI_Recv(m_synchBuf, m_MaxSynchStates, m_stateType, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		assert(status.MPI_TAG == (int)MasterCommTags::States || status.MPI_TAG == (int)MasterCommTags::LastStates);

		const bool rootIsGoal = Master::rootIsGoal(status.MPI_SOURCE);
		int nStates = 0;
		double start = MPI_Wtime();
		MPI_Get_count(&status, m_stateType, &nStates);
		
		// add states to the closed list
		for (int i = 0; i < nStates; i++) {
			Node<StateT, MoveT> *node = new Node<StateT, MoveT>(m_synchBuf[i]);

			// check for direct solution
			if (node->isTarget(rootIsGoal)) {
				// path from node to target										// SOLUTION FOUND
				if (node->getNumMoves() < pathLen) {
					assert(node->getNumMoves() <= maxSearchCost);

					cout << "Solution found in search (1): " << (int)node->getNumMoves() << endl;
					if (Verbose) {
						cout << "c(s) = d(s) + h(s) = " << (int)node->getCost(rootIsGoal) << " = " << (int)node->getNumMoves() << " + " << (int)node->getHeuristic(rootIsGoal) << endl;
					}

					if (rootIsGoal) {
						minimizePathLen(pathLen, minNodeG, minNodeS, node->getNumMoves(), *node, Node<StateT, MoveT>());
					} else {
						minimizePathLen(pathLen, minNodeG, minNodeS, node->getNumMoves(), Node<StateT, MoveT>(), *node);
					}
					maxSearchCost = min(maxSearchCost, pathLen);
				}
			} else {
				Node<StateT, MoveT> *found = m_closed[!rootIsGoal].find(node);

				if (found) {
					// found is a state with a known path to target				// SOLUTION FOUND
					//cout << *found << endl;
					//cout << *node << endl;
					const uint8_t newPathLen = found->getNumMoves() + node->getNumMoves();
					if (newPathLen < pathLen) {
						assert(newPathLen <= maxSearchCost);

						cout << "Solution found in search (2): " << (int)newPathLen << " (" << (int)node->getNumMoves() << " + " << (int)found->getNumMoves() << ")" << endl;
						if (Verbose) {
							cout << "c(s ) = d(s ) + h(s ) = " << (int)node->getCost(rootIsGoal) << " = " << (int)node->getNumMoves() << " + " << (int)node->getHeuristic(rootIsGoal) << endl;
							cout << "c(s') = d(s') + h(s') = " << (int)found->getCost(rootIsGoal) << " = " << (int)found->getNumMoves() << " + " << (int)found->getHeuristic(rootIsGoal) << endl;
						}

						if (rootIsGoal) {
							minimizePathLen(pathLen, minNodeG, minNodeS, newPathLen, *node, *found);
						} else {
							minimizePathLen(pathLen, minNodeG, minNodeS, newPathLen, *found, *node);
						}
						maxSearchCost = min(maxSearchCost, pathLen);
					}
				}
			}

			searchCost = node->getCost(rootIsGoal);

			// add node to the closed list
			if (!m_closed[rootIsGoal].addIfBetter(node)) {
				delete node;
			}
		}

		mTime += MPI_Wtime() - start;

		if (status.MPI_TAG == (int)MasterCommTags::States) {
			// inform searcher about maximum search cost
			MPI_Send(&maxSearchCost, 1, MPI_PathLen_Type, status.MPI_SOURCE, (int)MasterCommTags::MaxSearchCost, MPI_COMM_WORLD);
		} else if (status.MPI_TAG == (int)MasterCommTags::LastStates) {
			nLastStates++;
			running = nLastStates < nSearchers;
		} else {
			assert(false);
		}
	} while (running);

	//if (VeryVerbose) cout << "Master stops searcher" << endl;

	// finish searchers
	for (int p = 1; p < m_nProcs; p++) {
		MPI_Send(&running, 1, MPI_C_BOOL, p, (int)MasterCommTags::Finish, MPI_COMM_WORLD);
	}

	// compute average wall-clock times
	double time[NTimes] = { 0 }, gTime[NTimes];
	MPI_Reduce(time, gTime, NTimes, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	if (VeryVerbose) {
		cout << "Avg: search: " << gTime[0]/nSearchers << ", synch: " << gTime[1]/nSearchers << ", end: " << gTime[2]/nSearchers << ", TotCount: " << gTime[3] << endl;
		cout << "Master working time: " << mTime << endl;
	}

	//cout << "Counter = " << counter << endl;
	return buildPath(pathLen, minNodeG, minNodeS);
}

//////////////////////////////////////////////////////////////////////////////////////////////////
template<class StateT, class MoveT>
void Master<StateT, MoveT>::minimizePathLen(PathLenT& minPathLen, Node<StateT, MoveT>& minNodeG, Node<StateT, MoveT>& minNodeS, PathLenT pathLen, const Node<StateT, MoveT>& nodeG, const Node<StateT, MoveT>& nodeS) {
	if (pathLen < minPathLen || !minNodeG.isValid() && !minNodeS.isValid()) {
		// better solution found
		//cout << "path length improved from " << (int)minPathLen << " to " << (int)pathLen << endl;
		minNodeG = nodeG;
		minNodeS = nodeS;
		minPathLen = pathLen;
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////
// Build solution path.
// minNodeG: node with known path to goal node
// minNodeS: node with known path to start node
template<class StateT, class MoveT>
string Master<StateT, MoveT>::buildPath(PathLenT pathLen, Node<StateT, MoveT>& minNodeG, Node<StateT, MoveT>& minNodeS) {
	if (pathLen < NoSolution) {
		// solution found -> build solution path
		if (Verbose) cout << "build path of length = " << (int)pathLen << endl;

		if (!minNodeG.isValid()) {
			assert(minNodeS.isValid());
			// single path from minNodeS to start
			string result = buildPath(minNodeS, false, pathLen);
			assert(result.size() == pathLen);
			return result;
		} else if (!minNodeS.isValid()) {
			assert(minNodeG.isValid());
			// single path from minNodeG to goal
			string result = buildPath(minNodeG, true, pathLen);
			assert(result.size() == pathLen);
			return result;
		} else {
			// combined path
			assert(minNodeG.getState() == minNodeS.getState());
			string s1 = buildPath(minNodeS, false, pathLen);
			string s2 = buildPath(minNodeG, true, pathLen);
			
			s1.append(s2); // start to minNodeS, minNodeG to goal
			assert(s1.size() == pathLen);
			return s1;
		}

	} else {
		stringstream ss;
		ss << "NO SOLUTION WAS FOUND IN " << (int)m_maxMoves << " STEPS";
		return ss.str();
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////
// Build solution path from node v to goal or from start to v.
template<class StateT, class MoveT>
string Master<StateT, MoveT>::buildPath(const Node<StateT, MoveT>& v, bool rootIsGoal, int pathLen) {
	stringstream ss;

	if (rootIsGoal) {
		// from v to goal
		Node<StateT, MoveT> w = v;
		Node<StateT, MoveT> *node = m_closed[rootIsGoal].find(w.getState());

		while (node && !node->isRoot() && pathLen) {
			const MoveT m = Node<StateT, MoveT>::s_ReverseMoves[node->getMove()];
			ss << (int)m;
			w.apply(m);
			node = m_closed[rootIsGoal].find(w.getState());
			pathLen--;	// just to prevent endless-looping in case of an invalid path
		}
		if (VeryVerbose) cout << "build path from v to goal : " << ss.str() << " (" << ss.str().size() << ")" << endl;
		return ss.str();
	} else {
		// from start to v
		Node<StateT, MoveT> w = v;
		Node<StateT, MoveT> *node = m_closed[rootIsGoal].find(w.getState());

		while (node && !node->isRoot() && pathLen) {
			ss << (int)node->getMove();
			w.apply(Node<StateT, MoveT>::s_ReverseMoves[node->getMove()]);
			node = m_closed[rootIsGoal].find(w.getState());
			pathLen--;	// just to prevent endless-looping in case of an invalid path
		}
		string result = ss.str();
		reverse(result.begin(), result.end());
		if (VeryVerbose) cout << "build path from start to v: " << result << " (" << result.size() << ")" << endl;
		return result;
	}
}

