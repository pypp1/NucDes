**This readme is intended to be a guide in using the provided Python script.**

The code requires the _multiple_arrays.npz_ file to be placed in the same folder of the .py file itself in order to run. 

**If this condition is not met, the code will output an error.**

Once run, the code gives the user room to decide several parameters, like:

	• Internal and external pressure
   
	• The temperature used to compute heat transfer quantities
   
	• The adoption of approximations
   
	• Etc.

All of this happens through a series of prompts: when the code is executed, the user is prompted a series of questions which are to be answered using numbers. 

All acceptable answers are indicated in the questions themselves. **If any invalid answer is given by the user, the code will again output an error.**

Some questions may only appear based on previous answers. For example:

	• If the volumetric heat source q0 is considered, the user may consider the presence of a thermal shield or decide 
	  not to place it (unshielded case).

	• If the volumetric heat source q0 is not considered, the user is not prompted whether the thermal shield should be 
	  considered or not and the thermal shield is simply not considered.

At any point during its execution, the code can be aborted via a _KeyboardInterrupt_ input (_Ctrl+C_ or, if answering a question, _Ctrl+C and then Enter_). 

However, doing so will result in an incorrectly formatted output folder containing only part of the results. **It is thus recommended to follow through with the execution of the script.**

The code also contains a partially developed 2D simulation script, which computes the temperature and stress profiles along r and z. **Such section of the code is not complete and has been kept for future improvements.** 

While it will not give any error, it will only give partial results and **should not be used until completion.**

Additionally, after performing all the calculations, the user is given the possibility to display or not the computed plots. 

**All plots are now saved in the "Plots" folder, regardless of having been displayed or not.**

Finally, the code saves the results of each simulation in case-specific directories which, if not pre-existing, are created by the script itself. 

The name of the folders containing the results is built based on the simulation's parameters, so that all cases can be precisely distinguished, and each case's folder always contains:

	• A "Plots" folder
   
	• A "Final Results.txt" text file

The former contains all the saved plots, whereas the latter contains a recap of the simulation's chosen parameters and hypothesis and the related results. 

Everything contained in such text file is also displayed in the terminal for immediate consulting. The case directory path is always displayed at the end of each simulation.

This GitHub repository can be cloned to directly access the latest build. 

For any additional questions regarding the use of the code, feel free to contact us. Contributions and improvements are welcome!