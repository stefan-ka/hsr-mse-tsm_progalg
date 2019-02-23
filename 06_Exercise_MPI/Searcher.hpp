#pragma once

#include <sstream>
#include "Searcher.h"

//////////////////////////////////////////////////////////////////////////////////////////////////
// Add node to open list
template<class StateT, class MoveT>
bool Searcher<StateT, MoveT>::addNewNode(Node<StateT, MoveT> *node) {
	assert(node);
	
	// check if new state is already in the closed list
	Node<StateT, MoveT> *nodeC = m_closed.find(node);

	if (!nodeC || node < nodeC) {
		// check if new state is already in the open list
		Node<StateT, MoveT> *nodeO = m_open.find(node);

		if (!nodeO || node < nodeO) {
			// add new state to the open list
			if (nodeC) m_closed.erase(nodeC);
			if (nodeO) m_open.erase(nodeO);
			m_open.add(node);
			return true;
		}
	}
	return false;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
// Search: do at most MaxMoves/2 moves from root towards target state
template<class StateT, class MoveT>
void Searcher<StateT, MoveT>::search(const StateT& target) {
	bool running = true;
	PathLenT searchCost = 0, maxSearchCost = m_maxMoves;
	double start, time[NTimes] = { 0 };
	int flag = 0;
	MPI_Request synchRequest = MPI_REQUEST_NULL;

	//if (VeryVerbose) cout << "Worker search started" << endl;

	// do until all work has been done
	do {
		PathLenT searchDepth = 0;
		start = MPI_Wtime();

		// iterate through all open states
		if (!m_open.empty()) {
			Node<StateT, MoveT> * const current = m_open.removeFirst();
			assert(current);

			searchDepth = current->getNumMoves();
			searchCost = current->getCost();
			time[3]++;

			if (searchCost < maxSearchCost && searchDepth*2 < maxSearchCost) {
				// iterate through all possible moves
				for (int m = Node<StateT, MoveT>::s_FirstMove; m <= Node<StateT, MoveT>::s_LastMove; m++) {
					const MoveT move = MoveT(m);
					if (current->isPossible(move) && (current->isRoot() || !current->reversesLastMove(move))) {
						// first move or non-reverse move
						Node<StateT, MoveT> *newNode = new Node<StateT, MoveT>(current, move);

						if (!addNewNode(newNode)) {
							delete newNode; newNode = nullptr;
						}
					}
				}
			}

			//if (counter%10000 == 0) cout << "Cost = " << (int)searchCost << ", max search cost = " << (int)maxSearchCost << endl;

			// add current node to closed and synch list
			if (m_closed.addIfBetter(current)) {
				// it's the only place where synchStates are added
				m_synchStates.push_back(current->getState());
			} else {
				delete current;
			}

			//cout << "open: " << m_open.size() << ", openMap: " << m_open.size2() << ", closed: " << m_closed.size() << endl;
		}

		time[0] += MPI_Wtime() - start;

		
		// inter-process communication (synchronization)
		start = MPI_Wtime();

		communication(maxSearchCost, synchRequest);

		// test if still no work and final search cost reached
		//running = !m_open.empty() || searchCost < maxSearchCost;	
		running = !m_open.empty() || searchCost == 0;

		time[1] += MPI_Wtime() - start;

	} while (running);
	
	//if (VeryVerbose) cout << "Search finished" << endl;

	start = MPI_Wtime();

	// inform master about finished search
	int i = 0;
	while (!m_synchStates.empty()) {
		Node<StateT, MoveT> *node = m_closed.find(m_synchStates.front());
		if (node) m_synchBuf[i++] = *node;
		m_synchStates.pop_front();
	}
	// send closed states to master
	MPI_Send(m_synchBuf, i, m_stateType, 0, (int)MasterCommTags::LastStates, MPI_COMM_WORLD);

	//if (VeryVerbose) cout << "Master informed" << endl;

	// wait till all other searchers have stopped their search
	MPI_Request requests[2];
	bool dummy;

	// receive finish
	MPI_Irecv(&dummy, 1, MPI_C_BOOL, 0, (int)MasterCommTags::Finish, MPI_COMM_WORLD, &requests[0]);
	running = true;

	do {
		// receive work request
		MPI_Irecv(&dummy, 1, MPI_C_BOOL, MPI_ANY_SOURCE, (int)MasterCommTags::NeedsWork, m_group, &requests[1]);

		int idx = 0;
		MPI_Status status;
		MPI_Waitany(2, requests, &idx, &status);

		if (idx == 0) {
			// all other searcher have stopped their search
			//if (VeryVerbose) cout << "Finish received" << endl;
			running = false;
			MPI_Cancel(&requests[1]);
		} else {
			// reply to work request with no-open-states
			MPI_Send(m_workBuf, 0, m_stateType, status.MPI_SOURCE, (int)MasterCommTags::OfferWork, m_group);
			//if (VeryVerbose) cout << "Send no work" << endl;
		}
	} while (running);

	time[2] += MPI_Wtime() - start;

	if (VeryVerbose) cout << "Finished: search: " << time[0] << ", synch: " << time[1] << ", end: " << time[2] << ", cnt: " << time[3] << endl;

	// compute total times
	MPI_Reduce(time, nullptr, NTimes, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
}

//////////////////////////////////////////////////////////////////////////////////////////////////
// Interprocess communication
template<class StateT, class MoveT>
void Searcher<StateT, MoveT>::communication(PathLenT& maxSearchCost, MPI_Request& synchRequest) {
	const int groupSize = m_nProcs/2;

	MPI_Status status;
	int flag = 0;

	// check for maxium search cost from master
	MPI_Iprobe(0, (int)MasterCommTags::MaxSearchCost, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
	if (flag) {
		// get maximum search cost from master
		MPI_Recv(&maxSearchCost, 1, MPI_PathLen_Type, 0, (int)MasterCommTags::MaxSearchCost, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}

	// send newest closed states to master
	if (m_synchStates.size() == m_MaxSynchStates) {
		MPI_Wait(&synchRequest, MPI_STATUS_IGNORE); // protects m_synchBuf

		int i = 0;
		while (!m_synchStates.empty()) {
			Node<StateT, MoveT> *node = m_closed.find(m_synchStates.front());
			if (node) m_synchBuf[i++] = *node;
			m_synchStates.pop_front();
		}

		// send states to master
		MPI_Isend(m_synchBuf, i, m_stateType, 0, (int)MasterCommTags::States, MPI_COMM_WORLD, &synchRequest);
	}

	if (m_open.empty()) {
		// ask other processes in group for work (open states)
		int myGroupID;

		MPI_Comm_rank(m_group, &myGroupID);

		for (int i = 1; i < groupSize; i++) {
			const int p = (myGroupID + i)%groupSize;
			bool dummy;

			//if (VeryVerbose) cout << "send work request to " << (2 + 2*p - (m_myID & 1)) << endl;
			MPI_Isend(&dummy, 0, MPI_C_BOOL, p, (int)MasterCommTags::NeedsWork, m_group, &m_workRequests[p]);
		}

		// receive all messages until all work requests have been answered (blocking, because there is nothing else to do)
		int unAnswered = groupSize - 1;

		while (unAnswered) {
			MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, m_group, &status);
			if (status.MPI_TAG == (int)MasterCommTags::NeedsWork) {
				// no open states to offer
				bool dummy;
				MPI_Sendrecv(m_workBuf, 0, m_stateType, status.MPI_SOURCE, (int)MasterCommTags::OfferWork, &dummy, 1, MPI_C_BOOL, status.MPI_SOURCE, (int)MasterCommTags::NeedsWork, m_group, MPI_STATUS_IGNORE);

			} else if (status.MPI_TAG == (int)MasterCommTags::OfferWork) {
				int nOpenStates = 0;

				MPI_Recv(m_workBuf, m_MaxSynchStates, m_stateType, status.MPI_SOURCE, (int)MasterCommTags::OfferWork, m_group, &status);
				MPI_Get_count(&status, m_stateType, &nOpenStates);

				if (nOpenStates) {
					// get open states
					for (int i = 0; i < nOpenStates; i++) {
						Node<StateT, MoveT> *node = new Node<StateT, MoveT>(m_workBuf[i]);
						//cout << *node << endl;

						if (!addNewNode(node)) {
							delete node; node = nullptr;
						}
					}
				}
				MPI_Request_free(&m_workRequests[status.MPI_SOURCE]);
				unAnswered--;
			}
		}
		//if (VeryVerbose) cout << "open list size = " << m_open.size() << endl;

	} else {
		// check for work requests (non-blocking)
		MPI_Iprobe(MPI_ANY_SOURCE, (int)MasterCommTags::NeedsWork, m_group, &flag, &status);

		while (flag) {
			// work request is available: offer open.size/group.size open states
			const int nStates = min(m_MaxSynchStates, (int)m_open.size()/groupSize);

			// fill in work buffer
			for (int i = 0; i < nStates; i++) {
				Node<StateT, MoveT> *node = m_open.removeFirst();
				m_workBuf[i] = *node;
				delete node;
			}

			// receive work request and send work
			bool dummy;
			MPI_Sendrecv(m_workBuf, nStates, m_stateType, status.MPI_SOURCE, (int)MasterCommTags::OfferWork, &dummy, 1, MPI_C_BOOL, status.MPI_SOURCE, (int)MasterCommTags::NeedsWork, m_group, MPI_STATUS_IGNORE);

			// test for work request
			MPI_Iprobe(MPI_ANY_SOURCE, (int)MasterCommTags::NeedsWork, m_group, &flag, &status);
		}
	}
}

