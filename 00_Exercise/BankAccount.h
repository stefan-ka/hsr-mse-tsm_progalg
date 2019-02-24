#pragma once

#include "RWLock.h"
#include <chrono>

class BankAccount {
	mutable RWLock m_lock;	// mutable: can be modified even in const methods
	double m_balance = 0;	// bank account balance

public:
	void deposit(double amount) {
		m_lock.lockW();
		this_thread::sleep_for(chrono::milliseconds(100)); // just to force some concurrent reads
		m_balance = m_balance + amount;
		m_lock.unlockW();
	}

	double getBalance() const {
		double balance = 0;
		m_lock.lockR();
		this_thread::sleep_for(chrono::milliseconds(100)); // just to force some concurrent reads
		balance = m_balance;
		m_lock.unlockR();
		return balance;
	}

	size_t getReaders() const {
		return m_lock.getReaders();
	}
};