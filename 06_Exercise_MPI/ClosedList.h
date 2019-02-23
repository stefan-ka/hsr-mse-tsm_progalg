#pragma once

#include <unordered_map>
#include "Node.h"

//////////////////////////////////////////////////////////////////////////////////////////////////
// Seen and handled nodes
template<class StateT, class MoveT>
class ClosedList {
	unordered_map<StateT, Node<StateT, MoveT>*> m_closedMap;	

public:
	///////////////////////////////////////////////////////////////////////////
	~ClosedList() {
		for (auto v : m_closedMap) delete v.second;
	}

	///////////////////////////////////////////////////////////////////////////
	void add(Node<StateT, MoveT> *node) {
		assert(node);
		assert(!find(node));
		m_closedMap[node->getState()] = node;
	}

	///////////////////////////////////////////////////////////////////////////
	bool addIfBetter(Node<StateT, MoveT> *node) {
		assert(node);
		Node<StateT, MoveT> *found = find(node);

		if (!found || node->getNumMoves() < found->getNumMoves()) {
			m_closedMap[node->getState()] = node;
			delete found;
			return true;
		} else {
			return false;
		}
	}

	///////////////////////////////////////////////////////////////////////////
	void erase(Node<StateT, MoveT> *node) {
		assert(node);
		m_closedMap.erase(node->getState());
		assert(!find(node));
	}

	///////////////////////////////////////////////////////////////////////////
	Node<StateT, MoveT>* find(const StateT state) const {
		auto it = m_closedMap.find(state);

		return (it == m_closedMap.end()) ? nullptr : it->second;
	}

	///////////////////////////////////////////////////////////////////////////
	Node<StateT, MoveT>* find(Node<StateT, MoveT> *node) const {
		assert(node);
		auto it = m_closedMap.find(node->getState());

		return (it == m_closedMap.end()) ? nullptr : it->second;
	}
};