![](/home/miguel/Desktop/qiskit/qosf_2020/qosf.png)

------

# Quantum Computing Mentorship Program 								Fall 2020

In this repository, you will find my code submission to the QOSF Mentorship Program Fall 2020. The one I have decided to submit is #1. I decided to work in this particular task because I have a lot fun seeing how the optimization evolves tweaking variables and parameters. Sounds like a lot of fun right? 😝

For this version of the repository the <code>master branch</code> correspond precisely to Task #1. I do have the intention to solve the other task and add in their own branch for this same repository. So if you would like to see all the solutions (my solutions with my own style and approach) you have to stay tuned to this repository. 😎



🌟 Bonus feature: I am also planning to include a <code>julia</code> version of all the code, wherever possible of course. 





------

# Task #1: Problem Statement 

Implement, on a quantum simulator of your choice, the following 4 qubits state $\psi(\theta)$: 

![img](https://lh6.googleusercontent.com/xqdgbMQmz0_Vc0m7c8KD3kC6d6T5HtPcsdtFZRbDZ05TKQDLuWX5j40AjuuolfagOc8Tp0VWAx5MfFROt3j2pmFDo-7TM039M3sBsWG1fNIMODgDRydXbxw8Fwx-ScuTd2i4v_qG)

Where the number of layers, denoted with L, has to be considered as a parameter. We call ¨Layer¨ the combination of 1 yellow + 1 green block, so, for example, U1 + U2 is a layer. The odd/even variational blocks are given by:



Even blocks

![img](https://lh4.googleusercontent.com/MzwUCYyclCDhGXyEwfSN30AD5Xb1Q-MKqWkE08DTmCYADuymQDVO6yuUEFljXXJfcoQyX7nrDToJFBrM00hLkQhMDNhv5rBCFVpnng6Pc6eZL8hfuwHitCo1wj5ubEjtqx0-R45W)



Odd blocks

![img](https://lh4.googleusercontent.com/or2MXYRKkDp44ItUXlTB09c5GNipH7ihU3BOIXUuDV4jGrXg5CPLuGjVr6Aj8k4ir01jlSK4cCSOD9vbq-TUgDXSzwyuJ7t61emKROBOy-ARK9IJ1wxWP5onD6ZNo4FslYtrMv5X)



The angles $\theta_{i,n}$, are variational parameters, lying in the interval (0, 2), initialized at random. Double qubit gates are CZ gates.

Report with a plot, as a function of the number of layers, L, the minimum distance

$\epsilon = min_{\theta}  \norm( |\psi(\theta)> - |\phi> )$

Where $|\phi>$ is a randomly generated vector on 4 qubits and the norm $\norm(|v>) $, of a state | v>, simply denotes the square root of the sum of the modulus square of the components of |v >. The right set of parameters $\theta_{i,n}$ can be found via any method of choice (e.g. grid-search or gradient descent)

Bonus question:

Try using other gates for the parametrized gates and see what happens.





------

# Structure of My Code

I have included a <code>utils.py</code> file that contains all of the classes, functions, objects, and imports required for notebooks to run properly. 

As a user, you are just required to provide the parameters needs to initialize the simulations. For a detailed description of the parameters take a look at the Notebooks.



For Task #1, I have defined a <code>simulation</code> class that, among other parameters, receives the maximum number of layers that the user wants to include in the simulation. This class prepares the necessary circuits which are later executed. A <code>optimization</code> class has also been define to tune the variational parameters of the circuits. Proper handling and execution of all simulations is already provided within the Notebooks. 👍





# Requirements

Nothing out of the ordinary is required for you to run the Notebooks, just make sure you have qiskit and qiskit[visualization] installed. Regarding specific requirements for your OS you can check the following link: 

https://qiskit.org/documentation/install.html



# Acknowledgments

To the Quantum Open Source Foundation and its fantastic staff for organizing this event! 