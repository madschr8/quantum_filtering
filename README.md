# Quantum Filtering Using Physics-Informed Neural Networks
Python code made for a Bachelor Project in Physics at University of Copenhagen. Made by Lauge Lindegård, Thorbjørn Ramløv &amp; Mads Christian Olsen Barslev.

We use the $\texttt{QuTiP}$ library to generate quantum-noisy weak homodyne measurement records $J(t)$ for a qubit system subject to a Rabi-hamiltonian, measurement action, relaxation, and dephasing. Using a model consisting of an LSTM network and linear neural networks, we try to predict the underlying state evolution $\rho(t)$. The model is conditioned on the measurement record $J(t)$ and optimized only against a final projective measurement $y\in \{0,1\}$.

See the Bachlor Project pdf-file for more information.
