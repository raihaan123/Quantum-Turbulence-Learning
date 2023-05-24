17/5/23 - Teams
(Note - was postponed to 22/5/23 due to me being stuck in traffic...!)

### Milestones
- Normalized MFE component outputs to help RCMs - previously small components were mistaken for noise
- Also implemented the MFE modes - can now build the actual velocity profile
- CRCM looks very close to the desired output but needs slight tuning - testing W&B again but will launch a more rigorous experimental sweep
- QRCM has mixed results, looks in the right ballpark using manually tuned phase-encoding gains (essentially replacing the $4\pi$ term in literature with hyperparameters)
- I MAY have access to IBM Sherbrooke (127 qubits!!!) from a quantum challenge subscription - need to investigate how quicky I can place an experiment onto the queue
- Started the dissertation in parallel! Almost done with a first draft of the background, will send over 5 pages for feedback!


### Difficulties
- Tried rescaling $p, X$ and $b$ between 1 and -1, and then scaling the new hyperparamters, to control how much phase is added in each encoding step - but became very noisy - I feel very close to finding the correct tune but will probably need to leverage W&B again...
	- Taking a slight break for inspiration, instead focusing on the writeup at the moment, will migrate to a nicer Latex template soon!


### Questions
