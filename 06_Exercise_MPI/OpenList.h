#pragma once

#include <list>
#include <unordered_map>
#include "Node.h"

// Iterator Invalidation Rules
// http://kera.name/articles/2011/06/iterator-invalidation-rules-c0x/

//////////////////////////////////////////////////////////////////////////////////////////////////
// Efficient data structure for open nodes (states).
template<class StateT, class MoveT>
class OpenList {
	list<Node<StateT, MoveT>*> m_open;												// open nodes (states) sorted by costs
	unordered_map<StateT, typename list<Node<StateT, MoveT>*>::iterator> m_openMap;	// efficient search structure for open nodes

public:
	///////////////////////////////////////////////////////////////////////////
	~OpenList() {
		for (auto v : m_open) delete v;
	}

	///////////////////////////////////////////////////////////////////////////
	void add(Node<StateT, MoveT> *node) {
		assert(node);
		if (empty()) {
			m_open.push_back(node);
			m_openMap[node->getState()] = m_open.begin();
		} else {
			assert(!find(node));
			const auto beg = m_open.begin();
			const PathLenT costN = node->getCost();
			const PathLenT depthN = node->getNumMoves();

			auto it = --m_open.end();
			PathLenT costIt = (*it)->getCost();
			PathLenT depthIt = (*it)->getNumMoves();

			while (it != beg && (costN < costIt || costN == costIt && depthN < depthIt)) {
				--it;
				costIt = (*it)->getCost();
				depthIt = (*it)->getNumMoves();
			}
			if (it == beg) {
				if (costN < costIt || costN == costIt && depthN < depthIt) {
					// insert node as new front
					m_open.push_front(node);
					m_openMap[node->getState()] = m_open.begin();
				} else {
					// insert node after it
					m_openMap[node->getState()] = m_open.insert(++it, node);
				}
			} else {
				assert(costIt <= costN);
				// insert node after it
				m_openMap[node->getState()] = m_open.insert(++it, node);
			}
		}
		assert(*m_openMap[node->getState()] == node);
		assert(m_open.size() == m_openMap.size());
	}

	///////////////////////////////////////////////////////////////////////////
	// Erase node
	void erase(Node<StateT, MoveT> *node) {
		if (node) {
			auto it = m_openMap.find(node->getState());

			if (it != m_openMap.end()) {
				m_open.erase(it->second);
				m_openMap.erase(it);
				assert(m_open.size() == m_openMap.size());
			}
		}
	}

	///////////////////////////////////////////////////////////////////////////
	bool empty() const { return m_open.empty(); }

	///////////////////////////////////////////////////////////////////////////
	// Find node
	Node<StateT, MoveT>* find(Node<StateT, MoveT> *node) const {
		assert(node);
		auto it = m_openMap.find(node->getState());

		return (it == m_openMap.end()) ? nullptr : *it->second;
	}

	///////////////////////////////////////////////////////////////////////////
	// Get first open node
	Node<StateT, MoveT>* front() const { return m_open.front(); }

	///////////////////////////////////////////////////////////////////////////
	// Remove first open node
	Node<StateT, MoveT>* removeFirst() {
		assert(!m_open.empty());
		Node<StateT, MoveT>* node = m_open.front();

		m_open.pop_front();
		m_openMap.erase(node->getState());
		assert(m_open.size() == m_openMap.size());
		return node;
	}

	///////////////////////////////////////////////////////////////////////////
	size_t size() const { return m_open.size(); }
};

