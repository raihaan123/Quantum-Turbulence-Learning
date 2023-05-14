24/4/23 - Teams

### Milestones
- Implemented MFE solver in the PyRCM library - can output all 9 components as needed with RK4

Note - there is a discrepancy somewhere between the output from Alberta's implementation and mine - currently investigating, but reproducibility may be affected by use of different hardware. Either way shouldn't affect the training of the RCMs

### Questions
I can't demonstrate the usefulness of QRCMs by comparing execution times without significantly larger quantum computing access - Qiskit Aer is very slow on classical machines. Even though its apparent there will be a significant improvement with increasing qubit count (2^n states to leverage)

So in the report, I wanted to explore this other avenue...

- I want to show the usefulness of QRCMs by applying them to several intractable situations...
- I.e take the velocity/pressure/temperature field in the wake of arbitrary body!
- Either sourced from CFD or DNS... The point being that there is no neat set of ODEs to describe the structure of the solution, so integrating forward with RK4 needs nonlinear SPOD and ROM
- I want to train a CNN-based autoencoder to find a latent space - a 'pseudo'-SPOD - then train the RCM in the latent space

